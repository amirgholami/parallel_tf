import math
import time
import tensorflow as tf
import numpy as np
import multiprocessing as mp

import threading

from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/Users/yuconghe/Berkeley/FA17/distributed_tf/tmp/mnist_data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS


IMAGE_PIXELS = 28
def inference(x, ps_num):
  hid_w = tf.Variable(
      tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                          stddev=1.0 / IMAGE_PIXELS), name="hid_w")
  hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

# Variables of the softmax layer

  sm_w = tf.Variable(
      tf.truncated_normal([FLAGS.hidden_units, 10],
                          stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
      name="sm_w")
  sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

  hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
  hid = tf.nn.relu(hid_lin)

  y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

  return y

def run_model(job_name, task_index, barrier):
  # Create cluster:
  from tensorflow.examples.tutorials.mnist import input_data
  
  worker_hosts = ["localhost:22226", "localhost:22326", "localhost:22327", "localhost:22426"]
  cluster = tf.train.ClusterSpec({"ps": ["localhost:22225"],
                                "worker": worker_hosts})
  server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

  if job_name == "ps":
    server.join()

  num_workers = len(worker_hosts)

  is_chief = (task_index == 0)
  worker_device = "/job:worker/task:%d/cpu:0" % task_index

  step_and_accuracy = []


  with tf.device(tf.train.replica_device_setter(
    worker_device=worker_device,
    ps_device="/job:ps/cpu:0",
    cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)
    y_ = tf.placeholder(tf.float32, [None, 10])
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y = inference(x, task_index)
    loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    opt = tf.train.AdagradOptimizer(0.02)
    # Using synchronization. We can modify the source code to write our own
    # optimizer, but this one is very convenient.
    opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_workers,
                           total_num_replicas=num_workers)

    grads = opt.compute_gradients(loss)
    processed_grads = []
    for grad, var in grads:
      processed_grads.append((tf.multiply(grad, num_workers), var))

    train_op = opt.apply_gradients(processed_grads, global_step=global_step)
    summary_op = tf.summary.merge_all()

    # Need to run these tokens to start. Queues are used for synchronization.
    # see tf.train.SyncReplicasOptimizer
    local_init_op = opt.local_step_init_op
    if is_chief:
      local_init_op = opt.chief_init_op
    ready_for_local_init_op = opt.ready_for_local_init_op
    # Initial token and chief queue runners required by the sync_replicas mode
    chief_queue_runner = opt.get_chief_queue_runner()
    sync_init_op = opt.get_init_tokens_op()

    saver = tf.train.Saver()

    
    init_op = tf.global_variables_initializer()

    # Assigns ops to the local worker by default.
    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="./tmp/train_logs",
                             init_op=init_op,
                             local_init_op=local_init_op,
                             ready_for_local_init_op=ready_for_local_init_op,
                             global_step=global_step,
                             summary_op=summary_op,
                             recovery_wait_secs=0,
                             save_model_secs=600)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # print(mnist)


    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      device_filters=["/job:ps", "/job:worker/task:0", "/job:worker/task:1"])
    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
    if is_chief:
      # Chief worker will start the chief queue runner and call the init op.
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])
      # test_writer = tf.summary.FileWriter('./tmp/train_logs_summary', sess.graph)

    # Perform training.
    local_step = 0
    step = 0

    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                              y_: mnist.test.labels})
    print("Worker %d: training step %d done (global step: %d)" %
      (task_index, local_step, step))
    print("On trainer %d, iteration %d ps it reaches %f accuracy" % (task_index, step, test_accuracy))
    step_and_accuracy.append((step, test_accuracy))

    while not sv.should_stop() and step < 1000000:
      
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)

      # Setting for exactly same sgd
      for i in range(num_workers):
        if task_index == i:
          mini_batch_size = FLAGS.batch_size // num_workers
          start_index = int(mini_batch_size * i)
          end_index = int(mini_batch_size * (i+1))
          batch_xs = batch_xs[start_index:end_index]
          batch_ys = batch_ys[start_index:end_index]

      train_feed = {x: batch_xs, y_: batch_ys}
      _ = sess.run([train_op], feed_dict=train_feed)

      # Wait until all the processes finish this step.
      barrier.wait()

      step = sess.run([global_step])[0]
      local_step += 1

      if step % 100 == 0:
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                              y_: mnist.test.labels})
        print("Worker %d: training step %d done (global step: %d)" %
          (task_index, local_step, step))
        print("On trainer %d, iteration %d ps it reaches %f accuracy" % (task_index, step, test_accuracy))
        step_and_accuracy.append((step, test_accuracy))

      if step % 2000 == 0:
        print(step_and_accuracy)
  # Ask for all the services to stop.
  sv.stop()

def main(_):
    import random
    # Fix random seed to produce exactly the same results.
    random.seed(0)
    tf.set_random_seed(0)
    np.random.seed(0)

    threads = []
    b = threading.Barrier(parties=4)
    for i in range(4):
        threads.append(threading.Thread(target=run_model, args=("worker", i, b, )))
    threads.append(threading.Thread(target=run_model, args=("ps", 0, b, )))
    
    for t in threads:
        t.start()


if __name__ == "__main__":
  tf.app.run()
