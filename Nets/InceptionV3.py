import tensorflow as tf
import Config
import tensorflow.contrib.slim as slim
trunc_normal = lambda stddev:tf.truncated_normal_initializer(0.0, stddev)
class InceptionV3:
    def inception_v3_arg_scope(self, weight_dacay =4e-5, stddev = 0.1, batch_norm_var_collection = 'moving_vars'):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 0.001,
            'updates_collections':tf.GraphKeys.UPDATE_OPS,
            'variable_collections': {
                'beta':None,
                'gamma':None,
                'moving_mean':[batch_norm_var_collection],
                'moving_variance':[batch_norm_var_collection]
            }
        }
        with slim.arg_scope([slim.conv2d,slim.fully_connected],weights_regularizer=slim.l2_regularizer(weight_dacay)):
            with slim.arg_scope(
                [slim.conv2d],
                weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu,
                nomalizer_fn = slim.batch_norm,
                normalizer_params = batch_norm_params
            ) as sc:
                return sc

    def inception_v3_base(self, inputs, scope=None):
        end_points={}
        with tf.variable_scope(scope,'InceptionV3',[inputs]):
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='VALID'):
                net = slim.conv2d(inputs,32,[3,3],stride=2,scope='Conv2d_1a_33')
                net = slim.conv2d(net,32,[3,3],scope='Conv2d_2a_33')
                net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_33')
                net = slim.max_pool2d(net, [3,3], stride=2,scope='MaxPool_3a_33')
                net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_11')
                net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_33')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_33')

            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64,[1,1],scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_55')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_33')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_33')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3, 32, [1,1], scope='Conv2d_0b_11')
                    #12896=22432=256
                    net = tf.concat([branch_0, branch_1,branch_2,branch_3],3)
                    end_points['Mixed_5b']=net

                with tf.variable_scope('Mixed_5c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64,[1,1],scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_55')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_33')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_33')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_11')
                    #12896=22464=288
                    net = tf.concat([branch_0, branch_1,branch_2,branch_3],3)
                    end_points['Mixed_5c'] = net

                with tf.variable_scope('Mixed_5d'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64,[1,1],scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_55')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_33')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_33')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_11')
                    #12896=22464=288
                    net = tf.concat([branch_0, branch_1,branch_2,branch_3],3)
                    end_points['Mixed_5d'] = net

                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 384,[3,3],stride=2, padding='VALID',scope='Conv2d_1a_33')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_33')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_33')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3,3], stride=2,padding='VALID', scope='MaxPool_1a_33')
                    #38496288=768
                    net = tf.concat([branch_0, branch_1,branch_2],3)
                    end_points['Mixed_6a'] = net

                with tf.variable_scope('Mixed_6b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0a_17')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0a_71')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0b_17')
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_71')
                        branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_17')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0c_71')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_11')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    end_points['Mixed_6b'] = net

                with tf.variable_scope('Mixed_6c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0a_17')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0a_71')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0b_17')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_71')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_17')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0c_71')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_11')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    end_points['Mixed_6c'] = net

                with tf.variable_scope('Mixed_6d'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0a_17')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0a_71')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0b_17')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_71')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_17')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0c_71')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_11')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    end_points['Mixed_6d'] = net

                with tf.variable_scope('Mixed_6e'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0a_17')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0a_71')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0b_17')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_71')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_17')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0c_71')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope = 'AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_11')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                    end_points['Mixed_6e'] = net

                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_11')
                        branch_0 = slim.conv2d(branch_0, 320, [3,3], stride=2, padding='VALID',scope='Conv2d_1a_33')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0a_17')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0a_71')
                        branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_33')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3,3], stride=2,padding='VALID', scope='MaxPool_1a_33')
                    net = tf.concat([branch_0, branch_1,branch_2],3)
                    end_points['Mixed_7a'] = net

                with tf.variable_scope('Mixed_7b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = tf.concat(
                            [slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0a_13'),
                             slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0a_31')],3
                        )
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0a_33')
                        branch_2 = tf.concat(
                            [slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0a_13'),
                             slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0a_31')], 3
                        )
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3, 192, [1,1],scope = 'Conv2d_0a_11')

                    net = tf.concat([branch_0, branch_1,branch_2,branch_3],3)
                    end_points['Mixed_7b'] = net

                with tf.variable_scope('Mixed_7c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_11')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_11')
                        branch_1 = tf.concat(
                            [slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0a_13'),
                             slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0a_31')],3
                        )
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_11')
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0a_33')
                        branch_2 = tf.concat(
                            [slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0a_13'),
                             slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0a_31')], 3
                        )
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_33')
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0a_11')

                    net = tf.concat([branch_0, branch_1,branch_2,branch_3],3)
                    end_points['Mixed_7c'] = net
            return net, end_points

    def inference(self,inputs,
                     num_classes = Config.num_classes,
                     is_training = True,
                     dropout_keep_prob = 0.8,
                     prediction_fn = slim.softmax,
                     spatial_squeeze = True,
                     reuse = None,
                     scope = 'InceptionV3'
                     ):
        with tf.variable_scope(scope,'InceptionV3',[inputs,num_classes],reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
                net, end_points = self.inception_v3_base(inputs,scope=scope)
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d], stride=1,padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits,[5,5], stride=3, padding='VALID',scope='AvgPool_1a_55')
                    aux_logits = slim.conv2d(aux_logits,128,[1,1],scope='Conv_1b_11')
                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer=trunc_normal(0.01),
                                             padding='VALID', scope='Conv_2a_55')
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], weights_initializer=trunc_normal(0.001),
                                             activation_fn=None,normalizer_fn=None,scope='Conv_2b_11')
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits,[1,2],name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits
                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net, [8, 8],  padding='VALID', scope='AvgPool_1a_88')
                    net = slim.dropout(net,keep_prob=dropout_keep_prob,scope='Dropout_1b')
                    end_points['PreLogits'] = net
                    logits = slim.conv2d(net ,num_classes,[1,1], activation_fn=None,normalizer_fn=None,scope='Conv_1c_11')
                    if spatial_squeeze:
                        logits = tf.squeeze(logits,[1,2],name='SpatialSqueeze')
                    end_points['Logits']=logits
                    end_points['Predictions']=prediction_fn(logits,scope='Predictions')
        return end_points

    def loss(self,auxlogits,logits,labels):
        labels = tf.cast(labels,tf.int32)
        loss_auxlogits = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=auxlogits))
        loss_logits = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        loss = tf.add(loss_auxlogits,loss_logits)
        return loss_logits