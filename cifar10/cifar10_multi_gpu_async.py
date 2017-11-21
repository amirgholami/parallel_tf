from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import threading

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10
import cifar10_input
from cifar10_input import DataSet

IMAGE_SIZE = 24

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

def train(job_name, task_index, num_gpus=1, batch_size=128):
  """Train CIFAR-10 for a number of steps."""
  worker_hosts = ["localhost:22227"]
  server_hosts = ["localhost:22222"]

  cluster = tf.train.ClusterSpec({"ps": server_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

  logits = {}

  result = cifar10_input.load_CIFAR_batch(data_dir=os.path.join(os.getcwd(), 'tmp/cifar10_data'))
  cifar10_train = DataSet(images=result.train_X, labels=result.train_Y)
  cifar10_test = DataSet(images=result.test_X, labels=result.test_Y)

  # Calculate gradient separately for two gpus.
  # For async, we need only (one ps ... thus one cpu)
  
  gpu_idx = task_index % num_gpus
  cpu_idx = 0
  ps_idx = 0

  if job_name == "ps":
    server.join()

  # Put variables under the same variable scope for reuse.
  
  with tf.device(tf.train.replica_device_setter(
    ps_device='/job:ps/task:%d/cpu:%d' % (ps_idx, cpu_idx), # one ps_idx for async
    worker_device='/job:worker/task:%d/gpu:%d' % (task_index, gpu_idx))): # many task_idx for gpus


    """
    When you do tf.get_variable_scope().reuse_variables() you set the current 
    scope to reuse variables. If you call the optimizer in such scope, 
    it's trying to reuse slot variables, which it cannot find, 
    so it throws an error. If you put a scope around, 
    the tf.get_variable_scope().reuse_variables() only affects that scope, 
    so when you exit it, you're back in the non-reusing mode, the one you want.
    """

    with tf.variable_scope('local_ps_%d' % ps_idx):
      device_name = '%s_%d' % (cifar10.TOWER_NAME, task_index) # Device name for name_scope.
      with tf.name_scope(device_name) as scope:
          # Calculate the learning rate schedule
        num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size)
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)
        # Create a batch for each thread.
        raw_image_batch = tf.placeholder(tf.float32, shape=[batch_size // num_gpus, 32, 32, 3])
        label_batch = tf.placeholder(tf.int32, shape=[batch_size // num_gpus])
        image_batch = tf.image.resize_image_with_crop_or_pad(raw_image_batch, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
        loss, logit = tower_loss(scope, image_batch, label_batch, num_gpus)


        logits['logit_%d' % task_index] = logit
        tf.get_variable_scope().reuse_variables()
        grads = opt.compute_gradients(loss)

    # Calculate predictions accuracy
    top_k_op = tf.nn.in_top_k(logit, label_batch, 1)
    accuracy = tf.reduce_sum(tf.cast(top_k_op, tf.float32))

    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradients_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    # sess = tf.Session(config=tf.ConfigProto(
    #     allow_soft_placement=True,
    #     log_device_placement=False))
    # sess.run(init)
    # import pdb; pdb.set_trace()
    sv = tf.train.Supervisor(is_chief=(task_index == 0),
                      init_op=init,
                      logdir="./tmp",
                      recovery_wait_secs=0,
                      save_model_secs=600)
    # No local init...
    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)

    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    loss_lst = []
    accuracy_lst = []

    for step in xrange(1000000):
      start_time = time.time()

      #### Prepare input. ####
      feed_dict = {}
      images, labels = cifar10_train.next_batch(batch_size)

      mini_batch_size = batch_size // num_gpus
      start_index = int(mini_batch_size * task_index)
      end_index = int(mini_batch_size * (task_index+1))

      batch_xs = images[start_index:end_index]
      batch_ys = labels[start_index:end_index]
      feed_dict[raw_image_batch] = batch_xs
      feed_dict[label_batch] = batch_ys

      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      
      # print(grads[0][0])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

        loss_lst.append((step, loss_value))

      if step % 100 == 0:
        test_batches = 10000 // batch_size
        total = 0
        for i in xrange(test_batches):
          for j in xrange(num_gpus):
            test_images, test_labels = cifar10_test.next_batch(batch_size)
            test_dict = {}
            
            mini_batch_size = batch_size // num_gpus
            start_index = int(mini_batch_size * j)
            end_index = int(mini_batch_size * (j+1))
            batch_xs = test_images[start_index:end_index]
            batch_ys = test_labels[start_index:end_index]
            test_dict[raw_image_batch] = batch_xs 
            test_dict[label_batch] = batch_ys

          # print(test_dict)
          total += sess.run(accuracy, feed_dict=test_dict)
        print('step %d, accuracy: %.6f' % (step, total / 9984))
        accuracy_lst.append((step, total / 9984))


      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == 1000000:
        print(accuracy_lst)
        print(loss_lst)
        checkpoint_path = os.path.join('tmp/cifar10_train', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main():
  # Fix random seed to produce exactly the same results.
  import random
  random.seed(0)
  tf.set_random_seed(0)
  np.random.seed(0)
  num_gpus = 1
  threads = []
  for i in range(num_gpus):
    threads.append(threading.Thread(target=train, args=("worker", i, )))
  threads.append(threading.Thread(target=train, args=("ps", 0)))

  for t in threads:
    t.start()

if __name__ == "__main__":
  main()
      

