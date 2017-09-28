import math
import tensorflow as tf
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
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []

  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    # print(grad_and_vars)
    for g, _ in grad_and_vars:
      if g is not None:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

    # Average over the 'tower' dimension.
    if grads != []:
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)
  # print(average_grads)
  return average_grads

def accumulate_gradient_to_var(ps_num, average_grads):
    """
    Create a assign_op for given ps.
    Assign gradients to the copy on each ps in order to accumulate gradients.
    This is used to model communication step.
    """
    layer_1_vars = tf.get_collection(scope='copy_layer1_{0}'.format(ps_num), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    layer_2_vars = tf.get_collection(scope='copy_layer2_{0}'.format(ps_num), key=tf.GraphKeys.TRAINABLE_VARIABLES)

    all_vars = layer_1_vars + layer_2_vars

    # update_target_fn will be called everytime when new averaged_grads arrives at the ps.
    update_target_fn = []

    for tup in zip(average_grads, all_vars):
        grad, _ = tup[0]
        copy_target = tup[1]
        # add var to tensor
        update_target_fn.append(tf.assign_add(copy_target, grad))

    update_target_fn = tf.group(*update_target_fn)
    return update_target_fn

def update_var(ps_num):

    copied_layer_1_vars = tf.get_collection(scope='copy_layer1_{0}'.format(ps_num), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    copied_layer_2_vars = tf.get_collection(scope='copy_layer2_{0}'.format(ps_num), key=tf.GraphKeys.TRAINABLE_VARIABLES)

    copied_vars = copied_layer_1_vars + copied_layer_2_vars #

    target_layer_1_vars = tf.get_collection(scope='copy_layer1_1', key=tf.GraphKeys.TRAINABLE_VARIABLES)
    target_layer_2_vars = tf.get_collection(scope='copy_layer2_1', key=tf.GraphKeys.TRAINABLE_VARIABLES)

    target_vars = target_layer_1_vars + target_layer_2_vars

    graph_layer_1_vars = tf.get_collection(scope='layer1_{0}'.format(ps_num), key=tf.GraphKeys.TRAINABLE_VARIABLES)
    graph_layer_2_vars = tf.get_collection(scope='layer2_{0}'.format(ps_num), key=tf.GraphKeys.TRAINABLE_VARIABLES)

    graph_vars = graph_layer_1_vars + graph_layer_2_vars

    # update_target_fn will be called periodically to add accumulated grads in each ps group to center ps.
    update_target_fn = []

    # Update first
    for grad, target in zip(copied_vars, target_vars):
        update_target_fn.append(tf.assign_add(target, grad))
    # print(update_target_fn)

    # Zero out then
    for var in copied_vars:
        update_target_fn.append(tf.assign(
            var,
            tf.zeros(shape=var.shape)
        ))

    # Fetch variable thirdly.
    for source, target in zip(target_vars, graph_vars):
        update_target_fn.append(tf.assign(target, source))

    update_target_fn = tf.group(*update_target_fn)
    return update_target_fn

def inference(x, ps_num, is_copy=False):

  if is_copy:
    with tf.variable_scope('copy_layer1_{0}'.format(ps_num)) as scope:
        hid_w = tf.Variable(
            tf.zeros([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units]), name="hid_w")
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    with tf.variable_scope('copy_layer2_{0}'.format(ps_num)) as scope:
        sm_w = tf.Variable(
            tf.zeros([FLAGS.hidden_units, 10]),name="sm_w")
        sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

        return [hid_w, hid_b, sm_w, sm_b]

  else:
    with tf.variable_scope('layer1_{0}'.format(ps_num)) as scope:
        hid_w = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                stddev=1.0 / IMAGE_PIXELS), name="hid_w")
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    with tf.variable_scope('layer2_{0}'.format(ps_num)) as scope:
        sm_w = tf.Variable(
            tf.truncated_normal([FLAGS.hidden_units, 10],
                                stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
            name="sm_w")
        sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    with tf.variable_scope('softmax_{0}'.format(ps_num)) as scope:
        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

    return y

def main(_):
  # Create cluster:
  cluster = tf.train.ClusterSpec({"ps": ["localhost:22222", "localhost:22223"], "worker":["localhost:22888"]})
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  gradients = []

  with tf.device("/job:ps/task:0"):
      x0 = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y0_ = tf.placeholder(tf.float32, [None, 10])
      y0 = inference(x0, 0)
      global_step_0 = tf.Variable(0)

      accumulated_vars_0 = inference(None, 0, is_copy=True)

  with tf.device("/job:ps/task:1"):
      accumulated_vars_1 = inference(None, 1, is_copy=True)

  # Build the graph for two different worker, using the same params.
  for i in range(1):
      with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % i,
          cluster=cluster)):

        loss = -tf.reduce_sum(y0_ * tf.log(tf.clip_by_value(y0, 1e-10, 1.0)))

        correct_prediction = tf.equal(tf.argmax(y0_, 1), tf.argmax(y0, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        opt = tf.train.AdagradOptimizer(0.01)

        grad = opt.compute_gradients(loss)
        gradients.append(grad)

  with tf.device("/job:ps/task:0"):
    # Synchronize the grads
    average_grads_0 = average_gradients(gradients)

    # Apply the gradients to adjust the shared variables.
    train_op_0 = opt.apply_gradients(average_grads_0, global_step=global_step_0)
    accumulate_op_0 = accumulate_gradient_to_var(ps_num=0, average_grads=average_grads_0)
    update_op_0 = update_var(0) # Shall run it each 5 steps

  saver = tf.train.Saver()
  summary_op = tf.summary.merge_all()
  init_op = tf.initialize_all_variables()

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step_0,
                             save_model_secs=600)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      while not sv.should_stop() and step < 1000000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x0: batch_xs, y0_: batch_ys}

        sess.run(accumulate_op_0, feed_dict=train_feed)

        _, step = sess.run([train_op_0, global_step_0], feed_dict=train_feed)

        if step % 3 == 0:
            sess.run([update_op_0])

        if step % 100 == 0:
            # print(gradients)
            print( "Done step %d" % step)
            print("On iteration %d it reaches %f accuracy" % (step, sess.run(accuracy, feed_dict={x0: mnist.test.images,
                                                y0_: mnist.test.labels})))

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
