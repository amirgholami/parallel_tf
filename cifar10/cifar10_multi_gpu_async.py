from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10
import cifar10_input
from cifar10_input import DataSet

def tower_loss(scope, images, labels, num_gpus, batch_size=128):
  """Calculate the total loss on a single tower running the CIFAR model.
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  # print(images)
  logits = cifar10.inference(images, num_gpus)
  # print(logits, labels)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss, logits

def train(num_gpus=1, batch_size=128):
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		placeholders = {}
		for i in range(num_gpus):
			img_key = 'image_%d' % i
			placeholders[img_key] = tf.placeholder(tf.float32, shape=[batch_size // num_gpus, 32, 32, 3], name=img_key)
			label_key = 'label_%d' % i
			placeholders[label_key] = tf.placeholder(tf.int32, shape=[batch_size // num_gpus], name=label_key)
		def get_placeholder(name):
			return placeholders[name]

		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		# Calculate the learning rate schedule
		num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCHS_FOR_TRAIN / batch_size)
		decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

		# Decay the learning rate exponentially
		lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
																		global_step,
																		decay_steps,
																		cifar10.LEARNING_RATE_DECAY_FACTOR,
																		staircase=True)
		opt = tf.train.GradientDescentOptimizer(lr)
		result = cifar10_input.load_CIFAR_batch(data_dir=os.path.join(os.getcwd(), 'tmp/cifar10_data'))
		cifar10_train = DataSet(images=result.train_X, labels=result.train_Y)
		cifar10_test = DataSet(images=result.test_X, labels=result.test_Y)

		# Calculate gradient separately for two gpus.



