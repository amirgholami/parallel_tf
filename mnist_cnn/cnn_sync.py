import math
import os
# Fix random seed to produce exactly the same results.
import random
import numpy as np
import tensorflow as tf 
import threading

tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "tmp/mnist-data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=0)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference(x, ps_num):
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
        # G
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, seed=0)

        # Readout layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return keep_prob, y_conv

def run_model(job_name, task_index, barrier):
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)

    # Create cluster:
    from tensorflow.examples.tutorials.mnist import input_data

    is_chief = (task_index == 0)

    train_log_path = os.path.join(os.getcwd(), 'train_logs')

    worker_hosts = ["localhost:22329", "localhost:23329", "localhost:24329", "localhost:26329"]
    server_hosts = ["localhost:22322"]
    cluster = tf.train.ClusterSpec({"ps": server_hosts,
                                "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == "ps":
        server.join()

    num_workers = len(worker_hosts)

    step_and_accuracy = []

    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % task_index,
      ps_device="/job:ps/task:0",
      cluster=cluster)):
        
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        keep_prob, y = inference(x, 0)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        global_step = tf.contrib.framework.get_or_create_global_step()
        opt = tf.train.AdamOptimizer(1e-4)
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=len(worker_hosts), total_num_replicas=len(worker_hosts))
        grads = opt.compute_gradients(cross_entropy)
        # processed_grads = []
        # for grad, var in grads:
        #   processed_grads.append((tf.multiply(grad, 1), var))

        train_op = opt.apply_gradients(grads, global_step=global_step)

        local_init_op = opt.local_step_init_op
        if is_chief:
            local_init_op = opt.chief_init_op
        ready_for_local_init_op = opt.ready_for_local_init_op
        # Initial token and chief queue runners required by the sync_replicas mode
        chief_queue_runner = opt.get_chief_queue_runner()
        sync_init_op = opt.get_init_tokens_op()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir="./tmp/train_logs",
                             init_op=init_op,
                             local_init_op=local_init_op,
                             ready_for_local_init_op=ready_for_local_init_op,
                             global_step=global_step,
                             summary_op=summary_op,
                             recovery_wait_secs=0,
                             save_model_secs=600)

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        sess_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        # print("arriving...", task_index)
        if is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        local_step = 0
        step = 0

        # print("finished...")
        
        # test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
        #                                           y_: mnist.test.labels, keep_prob: 1.0})
        # print("Worker %d: training step %d done (global step: %d)" %
        #   (task_index, local_step, step))
        # print("On trainer %d, iteration %d ps it reaches %f accuracy" % (task_index, step, test_accuracy))
        # step_and_accuracy.append((step, test_accuracy))
        

        
        while not sv.should_stop() and step < 1000000:
            print(task_index, step)
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)

            # Setting for exactly same sgd
            for i in range(num_workers):
                if task_index == i:
                    mini_batch_size = FLAGS.batch_size // num_workers
                    start_index = int(mini_batch_size * i)
                    end_index = int(mini_batch_size * (i+1))
                    batch_xs = batch_xs[start_index:end_index]
                    batch_ys = batch_ys[start_index:end_index]

            _ = sess.run(train_op, feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1.0})
            step = sess.run(global_step)
            
            barrier.wait()
            local_step += 1

            # if step % 20 == 0 and step != 0:
            #     print("Worker %d: training step %d done (global step: %d)" % (task_index, local_step, step))
            

            if step % 100 == 0:
                test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                      y_: mnist.test.labels, keep_prob:1.0})
                print("Worker %d: training step %d done (global step: %d)" %
                  (task_index, local_step, step))
                print("On trainer %d, iteration %d ps it reaches %f accuracy" % (task_index, step, test_accuracy))
                step_and_accuracy.append((step, test_accuracy))

            if step % 2000 == 0:
                print(step_and_accuracy)

def main(_):
    from tensorflow.examples.tutorials.mnist import input_data

    threads = []
    b = threading.Barrier(parties=4)
    for i in range(4):
        threads.append(threading.Thread(target=run_model, args=("worker", i, b, )))

    threads.append(threading.Thread(target=run_model, args=("ps", 0, b, )))
    
    for t in threads:
        t.start()


if __name__ == "__main__":
  tf.app.run()