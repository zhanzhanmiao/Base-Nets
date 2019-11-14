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

# AlexNet and VGG
# Net = AlexNet()
Net = vggNet()
logits = Net.inference(x)
pred = tf.nn.softmax(logits)
loss = Net.loss(logits=logits, labels=y)

# Net = InceptionV3()
# endpoints = Net.inference(x)
# pred = endpoints['Predictions']
# logits = endpoints['Logits']
# loss = Net.loss(auxlogits=endpoints['AuxLogits'], logits=endpoints['Logits'],labels=y)

top_k_op = tf.nn.in_top_k(pred, y, 1)

# optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
train_step = optimizer.minimize(loss)

train_image, train_label = input_data.read_tfrecord(Config.tfrecord_file, BATCH_SIZE, shuffle=True, is_train=False)
test_image, test_label = input_data.read_tfrecord(Config.test_tfrecord_file, BATCH_SIZE, shuffle=False, is_train=False)

# sess = tf.Session()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
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


for epoch in range(EPOCH):
    for batch_num in range(BATCH_NUM):
        image_batch, label_batch = sess.run([train_image, train_label])
        loss_, pred_, logits_, top_k_op_ = sess.run([loss, pred, logits, top_k_op], feed_dict={x:image_batch, y:label_batch})
        training_precision = np.sum(top_k_op_)/BATCH_SIZE
        sess.run(train_step, feed_dict={x:image_batch, y:label_batch})
        if batch_num%10==0:
            print("epoch=%d, training_num=%d , loss=%.4f, training_precision=%.4f" % (epoch, batch_num, loss_, training_precision))

        if (batch_num*(epoch+1))%1000 == 0:
            checkpoint_path = os.path.join(Config.train_log_dir+'model.ckpt')
            saver.save(sess, checkpoint_path)

            test_batch_num = int(Config.test_num_samples/BATCH_SIZE)
            total_sample_count = test_batch_num*BATCH_SIZE
            true_count = 0
            for i in range(test_batch_num):
                test_image_, test_label_ = sess.run([test_image, test_label])

                predictions, pred_ = sess.run([top_k_op, pred], feed_dict={x:test_image_, y:test_label_})
                true_count += np.sum(predictions)

            precision = true_count / total_sample_count
            print("precision%.4f" % precision)



