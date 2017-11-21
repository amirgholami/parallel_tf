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

def inference(x, ps_num, is_copy=False):
    """
    Build the inference network
    """
    with tf.variable_scope('ps_%d' % ps_num) as scope:
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

def run_model(job_name, task_index):

    FLAGS.job_name = job_name
    FLAGS.task_index = task_index


    # Fix seed.
    tf.set_random_seed(seeds[FLAGS.task_index])
    np.random.seed(seeds[FLAGS.task_index])

    train_log_path = os.path.join(os.getcwd(), 'train_logs')

    worker_hosts = ["localhost:22329", "localhost:22330"]
    server_hosts = ["localhost:22322"]
    cluster = tf.train.ClusterSpec({"ps": server_hosts,
                                "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % FLAGS.task_index,
      ps_device="/job:ps/task:0",
      cluster=cluster)):
        
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        keep_prob, y = inference(x, 0)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        hooks=[tf.train.StopAtStepHook(last_step=1000000)]

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init_op = tf.global_variables_initializer()
        total_accuracy = []

    if FLAGS.job_name == "ps":
        server.join()

    with tf.train.MonitoredTrainingSession(
        master=server.target,
        is_chief=(FLAGS.task_index==0),
        checkpoint_dir=train_log_path,
        hooks=hooks) as sess:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        sess.run(init_op)
        for i in range(20000):
            batch = mnist.train.next_batch(FLAGS.batch_size // len(worker_hosts))
            sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
            if i % 5 == 0:
                print('step %d' % i)
            if i % 100 == 0 and i != 0:
                test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                print('step {0}, test accuracy {1}'.format(i, test_accuracy))
                total_accuracy.append(test_accuracy)
            if i % 2000 == 0:
                print(total_accuracy)

def main(_):
  Process(target=run_model, args=("ps", 0, )).start()

  Process(target=run_model, args=("worker", 0, )).start()
  Process(target=run_model, args=("worker", 1, )).start()
  # Process(target=run_model, args=("worker", 2, )).start()
  # Process(target=run_model, args=("worker", 3, )).start()

if __name__ == "__main__":
  tf.app.run()