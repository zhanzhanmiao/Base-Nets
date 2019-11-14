import tensorflow as tf
import os
import numpy as np
import input_data
from Nets.vggNet import vggNet
from Nets.AlexNet import AlexNet
from Nets.InceptionV3 import InceptionV3
import Config

IMG_WIDTH = Config.img_width
IMG_HEIGHT = Config.img_height
IMG_CHANNEL = Config.img_channel
BATCH_SIZE = Config.batch_size
LEARNING_RATE = Config.learning_rate
EPOCH = Config.epoch
BATCH_NUM = Config.batch_num

x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
y = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])

Net = vggNet(is_training=True)
logits = Net.inference(x)
pred = tf.nn.softmax(logits)

top_k_op = tf.nn.in_top_k(pred, y, 1)
test_image, test_label = input_data.read_tfrecord(Config.test_tfrecord_file, BATCH_SIZE, shuffle=False, is_train=False)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

saver = tf.train.Saver(tf.global_variables())
latest_ckpt = tf.train.latest_checkpoint(Config.train_log_dir)
graph = tf.get_default_graph()
if latest_ckpt is not None:
    saver.restore(sess, latest_ckpt)
else:
    init = tf.global_variables_initializer()
    sess.run(init)

coord = tf.train.Coordinator()
thread = tf.train.start_queue_runners(sess, coord)

test_batch_num = int(Config.test_num_samples/BATCH_SIZE)
total_sample_count = test_batch_num*BATCH_SIZE
true_count = 0
for i in range(test_batch_num):
    test_image_, test_label_ = sess.run([test_image, test_label])

    predictions, pred_ = sess.run([top_k_op, pred], feed_dict={x:test_image_, y:test_label_})
    true_count += np.sum(predictions)

precision = true_count / total_sample_count
print("precision%.4f" % precision)