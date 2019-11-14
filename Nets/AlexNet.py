import tensorflow as tf
import Config

class AlexNet:

    def print_activations(self, t):
        print(t.op.name, '', t.get_shape().as_list())

    def conv_block(self, x, input_dim, output_dim, kernel_size, stride, name, stddev=1e-1, trainable=True):
        with tf.name_scope(name) as scope, tf.variable_scope(name):
            kernel = tf.Variable(tf.truncated_normal([kernel_size,kernel_size,input_dim,output_dim], dtype=tf.float32, stddev=stddev, name='weights'))
            # kernel = tf.get_variable(name='weights', shape = [kernel_size, kernel_size,input_dim,output_dim],
            #                          dtype=tf.float32)#tf.contrib.layers.xavier_initializer_conv2d
            conv = tf.nn.conv2d(x, kernel, strides=[1,stride,stride,1],padding='SAME')
            biases = tf.Variable(tf.constant(0.0,shape=[output_dim], dtype=tf.float32),trainable=trainable, name='biases')
            conv = tf.nn.bias_add(conv, biases)
            conv = tf.layers.batch_normalization(conv, momentum=0.9,epsilon=1e-5, training=trainable, name='bn')
            conv1 = tf.nn.relu(conv, name=scope)

            return conv1

    def fc_block(self, x, input_dim, output_dim, name, is_lastfc=False):
        with tf.name_scope(name) as scope, tf.variable_scope(name):
            # weight = tf.Variable(tf.constant(0.1, tf.float32, [input_dim, output_dim]))
            weight = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[input_dim, output_dim], stddev=0.01))
            bias = tf.Variable(tf.constant(0.1, tf.float32, [output_dim]))
            output = tf.matmul(x, weight)+bias
            # if is_lastfc==False:
            output = tf.layers.batch_normalization(output, momentum=0.9, epsilon=1e-5, name='bn')
            output = tf.nn.relu(output, name=scope)
            return output



    def inference(self, x):
        conv1 = self.conv_block(x, 3, 64, 11, 4, 'conv1')
        self.print_activations(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        self.print_activations(pool1)

        conv2 = self.conv_block(pool1, 64, 192, 5, 1, 'conv2')
        self.print_activations(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        self.print_activations(pool2)

        conv3 = self.conv_block(pool2, 192, 384, 3, 1, 'conv3')
        self.print_activations(conv3)
        # pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        # self.print_activations(pool3)

        conv4 = self.conv_block(conv3, 384, 256, 3, 1, 'conv4')
        self.print_activations(conv4)
        # pool3 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        # self.print_activations(pool4)

        conv5 = self.conv_block(conv4, 256, 256, 3, 1, 'conv5')
        self.print_activations(conv4)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')


        pool5_reshape = tf.reshape(pool5, [Config.batch_size,-1])
        dim = pool5_reshape.get_shape()[1].value
        fc1 = self.fc_block(pool5_reshape, dim, 512, 'fc1')
        fc2 = self.fc_block(fc1, 512, 512, 'fc2')
        fc3 = self.fc_block(fc2, 512, Config.num_classes, 'fc3', is_lastfc=True)

        return fc3

    def loss(self, logits, labels):
        # labels = tf.cast(labels, tf.int64)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))