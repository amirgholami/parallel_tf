import math
import tensorflow as tf
import os
import time
import threading
import random
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

def apply_gradient_to_ps(ps_server_id, average_grads, opt):
    server_vars = tf.get_collection(scope='ps_{0}'.format(ps_server_id),key=tf.GraphKeys.TRAINABLE_VARIABLES)
    average_grads = [tup for tup in average_grads if tup[0] is not None]
    # update_target_fn will be called everytime when new averaged_grads arrives at the ps.
    update_target_fn = []
    for tup in zip(average_grads, server_vars):
        grad, _ = tup[0]
        var = tup[1]
        grad_and_var = (grad, var)
        update_target_fn.append(grad_and_var)
    return opt.apply_gradients(update_target_fn)

def update_var(ps_id, ps_server_id):
    local_vars = tf.get_collection(scope="ps_{0}".format(ps_id), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    ps_vars = tf.get_collection(scope="ps_{0}".format(ps_server_id), key=tf.GraphKeys.TRAINABLE_VARIABLES)  

    # Fetch variable.
    fetch_ps_fn = []
    for target, source in zip(local_vars, ps_vars):
        fetch_ps_fn.append(tf.assign(target, source))
    # Group into functions
    fetch_ps_fn = tf.group(*fetch_ps_fn)
    return fetch_ps_fn

def build_graph(x):
  hid_w = tf.Variable(
      tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                          stddev=1.0 / IMAGE_PIXELS), name="hid_w")
  hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
  sm_w = tf.Variable(
      tf.truncated_normal([FLAGS.hidden_units, 10],
                          stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
      name="sm_w")
  sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
  hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
  hid = tf.nn.relu(hid_lin)
  y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
  return y

def inference(x, ps_id, is_copy=False):
  with tf.variable_scope('ps_{0}'.format(ps_id)):
    y = build_graph(x)
  return y

def get_ps_task_id(idx, server_num):
  assert server_num >= 2, "Not well defined process groups."
  # Round robin.
  return idx % server_num

def run_model(job_name, task_index, barrier, lock):
  # Create cluster:
  train_log_path = os.path.join(os.getcwd(), 'train_logs')

  if task_index == 0 and job_name == "ps":
    import shutil
    if os.path.isdir(train_log_path):
      shutil.rmtree(train_log_path)

  worker_hosts = ["localhost:28427", "localhost:28428", "localhost:27427", "localhost:27428"]
  server_hosts = ["localhost:28422", "localhost:28423", "localhost:28334"]
  cluster = tf.train.ClusterSpec({"ps": server_hosts,
                                "worker": worker_hosts})
  server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

  worker_num = len(worker_hosts)
  # Use the last server as central ps.
  server_num = ps_server_id = len(server_hosts) - 1

  # Get ps id for this worker. (0,2) (1,3)
  ps_task_id = get_ps_task_id(task_index, server_num)

  group_size = worker_num // server_num
  is_group_chief = (task_index < server_num)
  is_chief = (task_index == 0)

  # Build central ps
  with tf.device("/job:ps/task:%d" % ps_server_id):
    ps_x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    ps_y_ = tf.placeholder(tf.float32, [None, 10])
    ps_y = inference(ps_x, ps_server_id)
    correct_prediction = tf.equal(tf.argmax(ps_y_, 1), tf.argmax(ps_y, 1))
    ps_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  with tf.device("/job:ps/task:%d" % ps_task_id):
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y = inference(x, ps_task_id)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    opt = tf.train.AdagradOptimizer(0.02)

    group_num_replicas = worker_num // server_num
    if task_index > group_num_replicas * server_num:
      group_num_replicas += 1
    sync_opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=group_num_replicas,
                           total_num_replicas=group_num_replicas)

    grads = sync_opt.compute_gradients(loss)

    processed_grads = []
    for grad, var in grads:
      if grad is not None:
        processed_grads.append((tf.multiply(grad, 1), var))
    
    train_op = sync_opt.apply_gradients(processed_grads, global_step=global_step)
    # 
    local_init_op = sync_opt.local_step_init_op

    if is_group_chief:
      local_init_op = sync_opt.chief_init_op
    ready_for_local_init_op = sync_opt.ready_for_local_init_op
    chief_queue_runner = sync_opt.get_chief_queue_runner()
    sync_init_op = sync_opt.get_init_tokens_op()

    # This step only carry out by the chief, after grad being computed.
    # accumulate_op = accumulate_gradient_to_var(ps_task_id, grad, opt)
    assign_to_ps_op = apply_gradient_to_ps(ps_server_id, processed_grads, opt)
    fetch_ps_op = update_var(ps_task_id, ps_server_id)
 
  if job_name == "ps":
    server.join()

  ###### Build the graph for different worker groups, using the same params. #####
  with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % task_index,
      ps_device="/job:ps/task:%d" % ps_task_id,
      cluster=cluster)):

    saver = tf.train.Saver()

    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    report = tf.report_uninitialized_variables()

  # Assigns ops to the local worker by default.
  # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_group_chief,
                             logdir=train_log_path,
                             local_init_op=local_init_op,
                             ready_for_local_init_op=ready_for_local_init_op,
                             global_step=global_step,
                             init_op=init_op,
                             summary_op=summary_op,
                             recovery_wait_secs=0,
                             save_model_secs=600)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    ps_specs = ["/job:ps/task:{0}".format(i) for i in range(len(server_hosts))]
    worker_specs = ["/job:worker/task:{0}".format(i) for i in range(worker_num)]

    specs = (ps_specs + worker_specs)

    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      device_filters=specs)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
    
    if is_group_chief:
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    # Loop until the supervisor shuts down or 1000000 steps have completed.
    step = 0
    local_step = 0
    step_and_accuracy = []
    
    while not sv.should_stop():
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)

      for i in range(worker_num):
        if task_index == i:
          mini_batch_size = FLAGS.batch_size // worker_num
          start_index = int(mini_batch_size * i)
          end_index = int(mini_batch_size * (i+1))
          batch_xs = batch_xs[start_index:end_index]
          batch_ys = batch_ys[start_index:end_index]

      train_feed = {x: batch_xs, y_: batch_ys}
      # _, step = sess.run([train_op, global_step], feed_dict=train_feed)
      # First wait: collect two gradients.
      sess.run(assign_to_ps_op, feed_dict=train_feed)
      barrier.wait()

      
      if is_group_chief:
        # Update shall be atomic. 
        # lock.acquire()
        sess.run([fetch_ps_op])
        # lock.release()

      # Second wait: wait for parameter to update.
      barrier.wait()
      
      local_step += 1
      step = sess.run(global_step)

      if local_step % 100 == 0:
        test_accuracy = sess.run(ps_accuracy, feed_dict={ps_x: mnist.test.images,
                                            ps_y_: mnist.test.labels})
        if is_chief:
          step_and_accuracy.append((local_step, test_accuracy))
        print("Worker %d: training step %d done (global step: %d)" %
          (task_index, local_step, step))
        print("On task %d On iteration %d ps it reaches %f accuracy" % (task_index, step, test_accuracy))
      
      if local_step % 2000 == 0 and is_chief:
        print(step_and_accuracy)

  # Ask for all the services to stop.
  sv.stop()

def main(_):
  import random
  # Fix random seed to produce exactly the same results.
  random.seed(0)

  threads = []
  b1 = threading.Barrier(parties=2)
  b2 = threading.Barrier(parties=2)
  l = threading.Lock()
  for i in [0, 2]:
      threads.append(threading.Thread(target=run_model, args=("worker", i, b1, l, )))
  threads.append(threading.Thread(target=run_model, args=("ps", 1, None, None, )))

  for i in [1, 3]:
      threads.append(threading.Thread(target=run_model, args=("worker", i, b2, l, )))
  threads.append(threading.Thread(target=run_model, args=("ps", 2, None, None, )))

  threads.append(threading.Thread(target=run_model, args=("ps", 0, None, None)))

  
  for t in threads:
      t.start()

if __name__ == "__main__":
  tf.app.run()
