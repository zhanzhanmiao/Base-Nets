import tensorflow as tf
import Config


class vggNet:
    def __init__(self,is_training):
        self.is_training = is_training

    def print_activations(self, t):
        print(t.op.name, '', t.get_shape().as_list())

    def conv_block(self, x, input_dim, output_dim, kernel_size, stride, name, stddev=1e-2, trainable=True):
        with tf.name_scope(name) as scope, tf.variable_scope(name):
            # kernel = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,input_dim,output_dim], dtype=tf.float32, stddev=stddev, name='weights'))
            kernel = tf.get_variable(name='weights', shape = [kernel_size, kernel_size,input_dim,output_dim],
                                     dtype=tf.float32)#tf.contrib.layers.xavier_initializer_conv2d
            conv = tf.nn.conv2d(x, kernel, strides=[1,stride,stride,1],padding='SAME')
            biases = tf.Variable(tf.constant(0.0,shape=[output_dim], dtype=tf.float32),trainable=trainable, name='biases')
            conv = tf.nn.bias_add(conv, biases)
            conv = tf.layers.batch_normalization(conv, trainable=self.is_training,training=self.is_training, name='bn')
            conv1 = tf.nn.relu6(conv, name=scope)

            return conv1

    def fc_block(self, x, input_dim, output_dim, name):
        with tf.name_scope(name) as scope, tf.variable_scope(name):
            weight = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[input_dim, output_dim], stddev=0.01))
            bias = tf.Variable(tf.constant(0.001, tf.float32, [output_dim]))
            output = tf.matmul(x, weight)+bias
            # if is_lastfc==False:
            output = tf.layers.batch_normalization(output, trainable=self.is_training, training=self.is_training, name='bn')
            output = tf.nn.relu6(output, name=scope)
            return output



    def inference(self, x):
        conv1_1 = self.conv_block(x, 3, 8, 3, 1, 'conv1_1')
        conv1_2 = self.conv_block(conv1_1, 8, 8, 3, 1, 'conv1_2')
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        self.print_activations(pool1)

        conv2_1 = self.conv_block(pool1, 8, 16, 3, 1, 'conv2_1')
        conv2_2 = self.conv_block(conv2_1, 16, 16, 3, 1, 'conv2_2')
        conv2_3 = self.conv_block(conv2_2, 16, 16, 3, 1, 'conv2_3')
        pool2 = tf.nn.max_pool(conv2_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        self.print_activations(pool2)

        conv3_1 = self.conv_block(pool2, 16, 32, 3, 1, 'conv3_1')
        conv3_2 = self.conv_block(conv3_1, 32, 32, 3, 1, 'conv3_2')
        conv3_3 = self.conv_block(conv3_2, 32, 32, 3, 1, 'conv3_3')
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
        self.print_activations(pool3)

        conv4_1 = self.conv_block(pool3, 32, 64, 3, 1, 'conv4_1')
        conv4_2 = self.conv_block(conv4_1, 64, 64, 3, 1, 'conv4_2')
        conv4_3 = self.conv_block(conv4_2, 64, 64, 3, 1, 'conv4_3')
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
        self.print_activations(pool4)

        conv5_1 = self.conv_block(pool4, 64, 128, 3, 1, 'conv5_1')
        conv5_2 = self.conv_block(conv5_1, 128, 128, 3, 1, 'conv5_2')
        conv5_3 = self.conv_block(conv5_2, 128, 128, 3, 1, 'conv5_3')
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool4')
        self.print_activations(pool5)

        conv6_1 = self.conv_block(pool5, 128, 256, 3, 1, 'conv6_1')
        conv6_2 = self.conv_block(conv6_1, 256, 256, 3, 1, 'conv6_2')
        conv6_3 = self.conv_block(conv6_2, 256, 256, 3, 1, 'conv6_3')
        pool6 = tf.nn.max_pool(conv6_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        self.print_activations(pool6)

        pool6_reshape = tf.reshape(pool6, [Config.batch_size,-1])
        dim = pool6_reshape.get_shape()[1].value
        fc1 = self.fc_block(pool6_reshape, dim, 1024, 'fc1')
        fc2 = self.fc_block(fc1, 1024, 4096, 'fc2')
        fc3 = self.fc_block(fc2, 4096, Config.num_classes, 'fc3')

        return fc3

    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
