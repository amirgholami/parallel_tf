import math
import tensorflow as tf
import os
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
tf.app.flags.DEFINE_integer("batch_size", 200, "Training batch size")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

def accumulate_gradient_to_var(ps_id, average_grads, opt):
    """
    Create a assign_op for given ps.
    Assign gradients to the copy on each ps in order to accumulate gradients.
    This is used to model communication step.
    """
    copied_local_vars = tf.get_collection(scope='copy_ps_{0}'.format(ps_id),key=tf.GraphKeys.TRAINABLE_VARIABLES)
    average_grads = [tup for tup in average_grads if tup[0] is not None]
    # update_target_fn will be called everytime when new averaged_grads arrives at the ps.
    update_target_fn = []
    for tup in zip(average_grads, copied_local_vars):
        grad, _ = tup[0]
        copy_target = tup[1]
        grad_and_var = (grad, copy_target)
        update_target_fn.append(grad_and_var)
    return opt.apply_gradients(update_target_fn)

def update_var(ps_id, ps_num):
    copied_local_vars = tf.get_collection(scope="copy_ps_{0}".format(ps_id), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    local_vars = tf.get_collection(scope="ps_{0}".format(ps_id), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    ps_vars = tf.get_collection(scope="ps_{0}".format(ps_num), key=tf.GraphKeys.TRAINABLE_VARIABLES)  
    # Update first.
    update_target_fn = []
    for grad, target in zip(copied_local_vars, ps_vars):
        update_target_fn.append(tf.assign_add(target, grad))
    # Zero out then
    zero_copy_fn = []
    for var in copied_local_vars:
        zero_copy_fn.append(tf.assign(
            var,
            tf.zeros(shape=var.shape)
        ))
    # Fetch variable thirdly.
    fetch_ps_fn = []
    for target, source in zip(local_vars, ps_vars):
        fetch_ps_fn.append(tf.assign(target, source))
    # Group into functions
    update_target_fn = tf.group(*update_target_fn)
    zero_copy_fn = tf.group(*zero_copy_fn)
    fetch_ps_fn = tf.group(*fetch_ps_fn)
    return update_target_fn, zero_copy_fn, fetch_ps_fn

def inference(x, ps_num, is_copy=False):
  ### Using copy instead of accumulator, since we don't need to sync here. ###
  if is_copy:
    with tf.variable_scope('copy_ps_{0}'.format(ps_num)) as scope:
      hid_w = tf.Variable(
          tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                              stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
      sm_w = tf.Variable(
          tf.truncated_normal([FLAGS.hidden_units, 10],
                              stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
          name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
      return (hid_w, hid_b, sm_w, sm_b)

  else:
    with tf.variable_scope('ps_{0}'.format(ps_num)) as scope:
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

def get_ps_task_id(idx, server_num):
  assert server_num >= 2, "Not well defined process groups."
  # Round robin.
  return idx % server_num

def main(_):
  # Create cluster:
  # worker_hosts = FLAGS.worker_hosts.split(",")
  # server_hosts = FLAGS.server_hosts.split(",")
  train_log_path = os.path.join(os.getcwd(), 'train_logs')
  print(train_log_path)

  if FLAGS.task_index == 0 and FLAGS.job_name == "ps":
    import shutil
    if os.path.isdir(train_log_path):
      shutil.rmtree(train_log_path)

  worker_hosts = ["localhost:22226", "localhost:22225", "localhost:22227", "localhost:22228"]
  server_hosts = ["localhost:22222", "localhost:22223", "localhost:22224"]
  cluster = tf.train.ClusterSpec({"ps": server_hosts,
                                "worker": worker_hosts})
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  worker_num = len(worker_hosts)
  # Use the last server as central ps.
  server_num = ps_server_id = len(server_hosts) - 1

  # Get ps id for this worker. (0,2) (1,3)
  ps_task_id = get_ps_task_id(FLAGS.task_index, server_num)

  group_size = worker_num // server_num
  is_group_chief = (FLAGS.task_index < group_size)
  is_chief = (FLAGS.task_index == 0)

  # Build central ps
  with tf.device("/job:ps/task:%d" % ps_server_id):
    ps_x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    ps_y_ = tf.placeholder(tf.float32, [None, 10])
    ps_y = inference(ps_x, ps_server_id)
    correct_prediction = tf.equal(tf.argmax(ps_y_, 1), tf.argmax(ps_y, 1))
    ps_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.device("/job:ps/task:0"):
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y = inference(x, 0)

    # Create copy to accumulate gradients.
    _ = inference(x, 0, is_copy=True) 

    global_step = tf.Variable(0, name="global_step", trainable=False)

    loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    opt = tf.train.AdagradOptimizer(0.01)

    group_num_replicas = worker_num // server_num
    if FLAGS.task_index > group_num_replicas * server_num:
      group_num_replicas += 1
    sync_opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=group_num_replicas,
                           total_num_replicas=group_num_replicas)
    # train_op = opt.minimize(loss, global_step=global_step)
    grad = sync_opt.compute_gradients(loss)
    
    if ps_task_id == 0:
      train_op = sync_opt.apply_gradients(grad, global_step=global_step)
      
      local_init_op = sync_opt.local_step_init_op
      if is_group_chief:
        local_init_op = sync_opt.chief_init_op
      ready_for_local_init_op = sync_opt.ready_for_local_init_op
      chief_queue_runner = sync_opt.get_chief_queue_runner()
      sync_init_op = sync_opt.get_init_tokens_op()

    # This step only carry out by the chief, after grad being computed.
    accumulate_op = accumulate_gradient_to_var(0, grad, opt)
    update_op, zero_copy_op, fetch_ps_op = update_var(ps_task_id, ps_server_id)


  with tf.device("/job:ps/task:1"):
    x1 = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y1_ = tf.placeholder(tf.float32, [None, 10])
    y1 = inference(x1, 1)

    # Create copy to accumulate gradients.
    _ = inference(x1, 1, is_copy=True) 

    global_step_1 = tf.Variable(0, name="global_step", trainable=False)

    loss_1 = -tf.reduce_sum(y1_ * tf.log(tf.clip_by_value(y1, 1e-10, 1.0)))

    correct_prediction_1 = tf.equal(tf.argmax(y1_, 1), tf.argmax(y1, 1))
    accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))

    opt_1 = tf.train.AdagradOptimizer(0.01)

    group_num_replicas = worker_num // server_num
    if FLAGS.task_index > group_num_replicas * server_num:
      group_num_replicas += 1
    sync_opt_1 = tf.train.SyncReplicasOptimizer(opt_1, replicas_to_aggregate=group_num_replicas,
                           total_num_replicas=group_num_replicas)
    # train_op = opt.minimize(loss, global_step=global_step)
    grad_1 = sync_opt_1.compute_gradients(loss_1)

    # Separate those local variable initialization from the global ones.
    if ps_task_id == 1:
      train_op_1 = sync_opt_1.apply_gradients(grad_1, global_step=global_step_1)
      
      local_init_op_1 = sync_opt_1.local_step_init_op
      if is_group_chief:
        local_init_op_1 = sync_opt_1.chief_init_op
      ready_for_local_init_op_1 = sync_opt_1.ready_for_local_init_op
      chief_queue_runner_1 = sync_opt_1.get_chief_queue_runner()
      sync_init_op_1 = sync_opt_1.get_init_tokens_op()

    accumulate_op_1 = accumulate_gradient_to_var(1, grad_1, opt_1)
    update_op_1, zero_copy_op_1, fetch_ps_op_1 = update_var(ps_task_id, ps_server_id)
 
  if FLAGS.job_name == "ps":
    server.join()

  ###### Build the graph for different worker groups, using the same params. #####
  with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % FLAGS.task_index,
      ps_device="/job:ps/task:%d" % ps_task_id,
      cluster=cluster)):

    if ps_task_id == 1:
      x = x1
      y = y1
      y_ = y1_
      global_step = global_step_1
      loss = loss_1
      correct_prediction = correct_prediction_1
      accuracy = accuracy_1
      opt = opt_1
      sync_opt = sync_opt_1
      grad = grad_1
      train_op = train_op_1
      accumulate_op = accumulate_op_1
      update_op = update_op_1
      zero_copy_op = zero_copy_op_1
      fetch_ps_op = fetch_ps_op_1
      local_init_op = local_init_op_1
      ready_for_local_init_op = ready_for_local_init_op_1
      chief_queue_runner = chief_queue_runner_1
      sync_init_op = sync_init_op_1

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
    
    while not sv.should_stop():
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size // worker_num)
      train_feed = {x: batch_xs, y_: batch_ys}

      _, step = sess.run([train_op, global_step], feed_dict=train_feed)

      local_step += 1

      sess.run([accumulate_op], feed_dict=train_feed)
      sess.run([update_op])
      sess.run([zero_copy_op])
      sess.run([fetch_ps_op])

      if step % 100 == 0:
          print("Worker %d: training step %d done (global step: %d)" %
            (FLAGS.task_index, local_step, step))
          print("On task %d On iteration %d ps it reaches %f accuracy" % (FLAGS.task_index, step, sess.run(ps_accuracy, feed_dict={ps_x: mnist.test.images,
                                              ps_y_: mnist.test.labels})))

  # Ask for all the services to stop.
  sv.stop()

if __name__ == "__main__":
  tf.app.run()
