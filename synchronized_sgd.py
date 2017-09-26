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
    #   else:
    #     expanded_g = tf.expand_dims(tf.Variable(0), 0)
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


def inference(x, ps_num):

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
  cluster = tf.train.ClusterSpec({"ps": ["localhost:22222"], "worker":["localhost:22888", "localhost:22889", "localhost:22890"]})
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

  gradients = []

  with tf.device("/job:ps/task:0"):
      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])
      y = inference(x, 0)

  # Build the graph for two different worker, using the same params.
  for i in range(3):
      with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % i,
          cluster=cluster)):

        loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        global_step = tf.Variable(0)

        opt = tf.train.AdagradOptimizer(0.01)

        grad = opt.compute_gradients(loss)
        gradients.append(grad)

  # Synchronize the grads
  average_grads = average_gradients(gradients)

  # Apply the gradients to adjust the shared variables.
  train_op = opt.apply_gradients(average_grads, global_step=global_step)

  # train_op = opt.minimize(loss, global_step=global_step)

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
                             global_step=global_step,
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
        train_feed = {x: batch_xs, y_: batch_ys}

        _, step = sess.run([train_op, global_step], feed_dict=train_feed)
        if step % 100 == 0:
            # print(gradients)
            print( "Done step %d" % step)
            print("On iteration %d it reaches %f accuracy" % (step, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                y_: mnist.test.labels})))

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
