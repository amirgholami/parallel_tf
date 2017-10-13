import math
import tensorflow as tf 
import os
import numpy as np
from multiprocessing import Process, Lock
from tensorflow.examples.tutorials.mnist import input_data

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

seeds = [1,2,3,4,5,6]

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


def accumulate_gradient_to_var(ps_id, average_grads, opt, global_step):
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
    return opt.apply_gradients(update_target_fn, global_step=global_step)


def update_var(ps_id, ps_num):
    copied_local_vars = tf.get_collection(scope="copy_ps_{0}".format(ps_id), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    local_vars = tf.get_collection(scope="ps_{0}".format(ps_id), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    ps_vars = tf.get_collection(scope="ps_{0}".format(ps_num), key=tf.GraphKeys.TRAINABLE_VARIABLES)  
    # Update first.
    update_target_fn = []
    for grad, target in zip(copied_local_vars, ps_vars):
        update_target_fn.append(tf.assign_add(target, grad, use_locking=True))
    # Zero out then
    zero_copy_fn = []
    for var in copied_local_vars:
        zero_copy_fn.append(tf.assign(
            var,
            tf.zeros(shape=var.shape),
            use_locking=True
        ))
    # Fetch variable thirdly.
    fetch_ps_fn = []
    for target, source in zip(local_vars, ps_vars):
        fetch_ps_fn.append(tf.assign(target, source, use_locking=True))
    # Group into functions
    update_target_fn = tf.group(*update_target_fn)
    zero_copy_fn = tf.group(*zero_copy_fn)
    fetch_ps_fn = tf.group(*fetch_ps_fn)
    return update_target_fn, zero_copy_fn, fetch_ps_fn

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def build_graph(x):
  """
  Build the inference network
  """
  # The first two dimensions are the patch size, 
  # the next is the number of input channels, 
  # and the last is the number of output channels. 
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])

  # To apply the layer, we first reshape x to a 4d tensor, 
  # with the second and third dimensions corresponding to image width and height, 
  # and the final dimension corresponding to the number of color channels.
  x = tf.reshape(x, [-1, 28, 28, 1])
  h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)


  # In order to build a deep network, we stack several layers of this type.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # Now that the image size has been reduced to 7x7, 
  # we add a fully-connected layer with 1024 neurons to allow processing on the entire image. 
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Add dropout
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Readout layer
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return keep_prob, y_conv

def inference(x, ps_id, is_copy=False):
  if is_copy:
    with tf.variable_scope('copy_ps_{0}'.format(ps_id)):
      keep_prob, y_conv = build_graph(x)
  else:
    with tf.variable_scope('ps_{0}'.format(ps_id)):
      keep_prob, y_conv = build_graph(x)
  return keep_prob, y_conv

def get_ps_task_id(worker_id, server_num):
  assert server_num >= 2, "Not well defined process groups."
  # Round robin.
  return worker_id % server_num

def run_model(lock, job_name, task_index):

  FLAGS.job_name = job_name
  FLAGS.task_index = task_index

  # Fix seed.
  tf.set_random_seed(seeds[FLAGS.task_index])
  np.random.seed(seeds[FLAGS.task_index])

  ####### Create cluster ########
  train_log_path = os.path.join(os.getcwd(), 'train_logs')
  if FLAGS.task_index == 0 and FLAGS.job_name == "ps":
    import shutil
    if os.path.isdir(train_log_path):
      shutil.rmtree(train_log_path)
    

  worker_hosts = ["localhost:22227", "localhost:22228", "localhost:22229", "localhost:22230"]
  server_hosts = ["localhost:22222", "localhost:22223", "localhost:22224"]

  cluster = tf.train.ClusterSpec({"ps": server_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  worker_num = len(worker_hosts)

  # Use the last server as central ps.
  server_num = ps_server_id = len(server_hosts) - 1

  # Get ps id for this worker. (0,2) (1,3)
  ps_id = get_ps_task_id(FLAGS.task_index, server_num)
  group_size = worker_num // server_num
  is_group_chief = (FLAGS.task_index < server_num)
  is_chief = (FLAGS.task_index == 0)

  # print(ps_id, group_size, FLAGS.task_index, is_group_chief)

  # Build central ps.
  with tf.device("/job:ps/task:%d" % ps_server_id):
    ps_x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS * IMAGE_PIXELS])
    ps_y_ = tf.placeholder(tf.float32, shape=[None, 10])
    ps_keep_prob, ps_y = inference(ps_x, ps_server_id)
    correct_prediction = tf.equal(tf.argmax(ps_y_, 1), tf.argmax(ps_y, 1))
    ps_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Build graph on each task.
  with tf.device("/job:ps/task:%d" % ps_id):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    keep_prob, y = inference(x, ps_id)

    # Create copy to accumulate gradients.
    _ = inference(x, ps_id, is_copy=True)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    opt = tf.train.AdamOptimizer(1e-4, use_locking=True)
    group_num_replicas = worker_num // server_num

    # Assign extra workers if not divisible.
    print(group_num_replicas)
    if FLAGS.task_index > group_num_replicas * server_num:
      group_num_replicas += 1

    sync_opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=group_num_replicas,
                       total_num_replicas=group_num_replicas, use_locking=True)
    grad = sync_opt.compute_gradients(loss)
    train_op = sync_opt.apply_gradients(grad, global_step=global_step)

    local_init_op = sync_opt.local_step_init_op
    if is_group_chief:
        local_init_op = sync_opt.chief_init_op
    ready_for_local_init_op = sync_opt.ready_for_local_init_op
    chief_queue_runner = sync_opt.get_chief_queue_runner()
    sync_init_op = sync_opt.get_init_tokens_op()

    # This step only carry out by the chief, after grad being computed.
    # accumulate_op = accumulate_gradient_to_var(ps_id, grad, opt, None)
    update_op, zero_copy_op, fetch_ps_op = update_var(ps_id, ps_server_id)
    assign_to_ps_op = apply_gradient_to_ps(ps_server_id, grad, opt)

    report = tf.report_uninitialized_variables()

  if FLAGS.job_name == "ps":
    server.join()

  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    ps_device="/job:ps/task:%d" % ps_id, 
    cluster=cluster)):
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

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

    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    if is_group_chief:
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    step = 0
    local_step = 0
    step_and_accuracy = []

    while not sv.should_stop():
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size // worker_num)
      train_feed = {x: batch_xs, y_: batch_ys, keep_prob:0.5}

      _, step = sess.run([train_op, global_step], feed_dict=train_feed)

      local_step += 1

      if is_group_chief:
        # Update shall be atomic. 
        lock.acquire()
        sess.run(assign_to_ps_op, feed_dict=train_feed)
        sess.run([fetch_ps_op])
        lock.release()
        

      if step % 5 == 0:
        print("Worker %d: training step %d done (global step: %d)" % (FLAGS.task_index, local_step, step))
        train_accuracy = sess.run(ps_accuracy, feed_dict={ps_x: batch_xs,
                                            ps_y_: batch_ys, ps_keep_prob:1.0})
        print("On task %d On iteration %d ps it reaches %f accuracy" % (FLAGS.task_index, step, train_accuracy))
        
        
      if step % 100 == 0 and step != 0:
        lock.acquire()
        # While computing, shall be locked.
        test_accuracy = sess.run(ps_accuracy, feed_dict={ps_x: mnist.test.images,
                                            ps_y_: mnist.test.labels, ps_keep_prob:1.0})
        lock.release()
        print("On task %d On iteration %d ps it reaches %f accuracy" % (FLAGS.task_index, step, test_accuracy))
        step_and_accuracy.append((step, test_accuracy))
      if step % 2000 == 0 and is_chief:
        print(step_and_accuracy)

def main():

  lock = Lock()
  lock = Lock()
  Process(target=run_model, args=(lock, "ps", 0, )).start()
  Process(target=run_model, args=(lock, "ps", 1, )).start()
  Process(target=run_model, args=(lock, "ps", 2, )).start()

  Process(target=run_model, args=(lock, "worker", 0, )).start()
  Process(target=run_model, args=(lock, "worker", 1, )).start()
  Process(target=run_model, args=(lock, "worker", 2, )).start()
  Process(target=run_model, args=(lock, "worker", 3, )).start()


if __name__ == "__main__":
  main()