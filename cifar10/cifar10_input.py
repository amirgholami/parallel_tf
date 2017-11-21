import tensorflow as tf
import numpy as np
import pickle
import os

import urllib
import sys
import tarfile

from six.moves import xrange

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

class DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype=tf.float32):
    """Construct a DataSet.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def load_CIFAR_batch(data_dir):
  """ 
  load cifar-10 into memory, instead of Queue approach.
  we need to get rid of randomness.
  """

  class Cifar10Dataset(object):
    pass

  result = Cifar10Dataset()
  # # Download cifar-10 data
  maybe_download_and_extract(data_dir)
  data_dir = os.path.join(data_dir, 'cifar-10-batches-py')

  # # First construct filename lists
  filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  test_filename = os.path.join(data_dir, 'test_batch')

  train_X = []
  train_Y = []
  test_X = []
  test_Y = []

  for filename in filenames:
    with open(filename, 'rb') as f:
      datadict = pickle.load(f, encoding='bytes')
      train_X.append(datadict[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")) # 10000 * 3072 -> 10000 * 3 * 32 * 32
      train_Y.append(np.array(datadict[b'labels']))

  with open(test_filename, 'rb') as f:
    datadict = pickle.load(f, encoding='bytes')
    test_X.append(datadict[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float"))
    test_Y.append(np.array(datadict[b'labels']))

  #### These are another version ####
  # train_X = np.concatenate(train_X, axis=0)
  # train_Y = np.concatenate(train_Y, axis=0)
  # test_X = np.concatenate(test_X, axis=0)
  # test_Y = np.concatenate(test_Y, axis=0)
  # # import pdb; pdb.set_trace()

  # # [5000, 3, 32, 32] * 10
      
  
  # result.train_X = train_X
  # result.test_X = test_X
  # result.train_Y = train_Y
  # result.test_Y = test_Y
  # return result

  # def create_standard_tensors(images, distorted=False):
  #   print("Spawning 10 threads to resize images... make take some time.")
  #   import threading
  #   WORKER_COUNT = 1
  #   ret = [[] for i in range(WORKER_COUNT)]
  #   threads = []


  #   def worker_func(imgs, lst, thread_idx, distorted=False):
  #     height = IMAGE_SIZE
  #     width = IMAGE_SIZE
  #     for i in range(len(imgs)):
  #       img = imgs[i]
  #       if i % 10 == 0:
  #         print("Thread_idx: %d done %3f" % (thread_idx, i / len(imgs)))
  #       if distorted:
  #         pass
  #       else:
  #         resized_image = tf.image.resize_image_with_crop_or_pad(img, height, width)
  #         float_image = tf.image.per_image_standardization(img)
  #         lst.append(float_image)
  #     return


  #   for i in range(WORKER_COUNT):
  #     mini_batch_size = len(images) // WORKER_COUNT
  #     batch_images = images[mini_batch_size * i: mini_batch_size * (i+1)]
  #     t = threading.Thread(target=worker_func, args=(batch_images, ret[i], i, ))
  #     threads.append(t)
  #     t.start()

  #   for t in threads:
  #     t.join()
      
  #   return ret



  # Now stack those arrays
  ############# These are personal standardize functions ##############
  
  train_X = np.concatenate(train_X, axis=0).reshape(50000, -1)
  train_Y = np.concatenate(train_Y, axis=0)
  test_X = np.concatenate(test_X, axis=0).reshape(10000, -1)
  test_Y = np.concatenate(test_Y, axis=0)

  def standardize(X):
    """
    Implementation of tf.image.per_image_standardization
    """ 
    means = (X - np.mean(X, axis=1, keepdims=True))
    adj_stds = np.maximum(np.std(X, axis=1), 1.0/np.sqrt(32 * 32 * 3))
    return means / adj_stds.reshape(adj_stds.shape[0], 1)

  # [50000, 3072]
  result.train_X = standardize(train_X).reshape(50000, 32, 32, 3)
  result.test_X = standardize(test_X).reshape(10000, 32, 32, 3)
  result.train_Y = train_Y
  result.test_Y = test_Y
  return result
  ############# These are personal standardize functions ##############

def maybe_download_and_extract(data_dir):
  """Download and extract the tarball from Alex's website."""
  dest_directory = data_dir

  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

load_CIFAR_batch("./tmp/cifar10_data")
