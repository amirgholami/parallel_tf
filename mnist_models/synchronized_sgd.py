
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def build_mnist_model(x):

  # with tf.device("/cpu:0"):
      # layer 1
  # Need to be reuse among different processes.
  with tf.variable_scope('layer1'):
      W1 = tf.get_variable('w1',[784,200],
                         initializer=tf.zeros_initializer())
      b1 = tf.get_variable('b1',[200,],
                         initializer=tf.constant_initializer(0.0))

      y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

# layer 2
  with tf.variable_scope('layer2'):
      W2 = tf.get_variable('w2',[200,10],
                     initializer= tf.random_normal_initializer())

      b2 = tf.get_variable('b2',[10,],
                         initializer=tf.constant_initializer(0.0))
      y2 = tf.matmul(y1, W2) + b2

  # output
  y = y2

  return y

def reuse_mnist_model(x):
    with tf.variable_scope('layer1', reuse=True):
        W1 = tf.get_variable('w1')
        b1 = tf.get_variable('b1')
        y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    with tf.variable_scope('layer2', reuse=True):
        W2 = tf.get_variable('w2')
        b2 = tf.get_variable('b2')
        y = tf.matmul(y1, W2) + b2
    return y


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model, build the queue. It is used to manage asynchronous update.
  input_images = tf.placeholder(tf.float32, [None, 784])
  input_labels = tf.placeholder(tf.float32, [None, 10])

  queue = tf.FIFOQueue(capacity=500, dtypes=[tf.float32, tf.float32], shapes=[[784], [10]])
  enqueue_op = queue.enqueue_many([input_images, input_labels])
  dequeue_op = queue.dequeue()

  x, y_ = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
  # print(x, y_)

  y = build_mnist_model(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  def enqueue(sess):
      """ Iterates over our data puts small junks into our queue."""
      while True:
        batch_xs, batch_ys = mnist.train.next_batch(30)
        sess.run(enqueue_op, feed_dict={input_images: batch_xs,
                                        input_labels: batch_ys})
        # print("added to the queue")
      print("finished enqueueing")

  enqueue_thread = threading.Thread(target=enqueue, args=[sess])
  enqueue_thread.isDaemon()
  enqueue_thread.start()

  # Create a coordinator, launch the queue runner threads.
  coord = tf.train.Coordinator()
  enqueue_threads = tf.train.start_queue_runners(sess, coord=coord)

  # Test trained model
  eval_y = reuse_mnist_model(input_images)

  # Train
  for i in range(10000):
    # print(i)
    import pdb; pdb.set_trace()
    sess.run(train_step)


    if i % 100 == 0:

      correct_prediction = tf.equal(tf.argmax(eval_y, 1), tf.argmax(input_labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print("On iteration %d it reaches %f accuracy" % (i, sess.run(accuracy, feed_dict={input_images: mnist.test.images,
                                          input_labels: mnist.test.labels})))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
