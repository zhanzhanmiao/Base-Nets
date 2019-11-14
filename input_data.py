import tensorflow as tf
import Config
import cv2

def read_tfrecord(tfrecord_file, batch_size, shuffle, is_train=True):
    data_files = tf.gfile.Glob(tfrecord_file)
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                        'label':tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image,[Config.img_height, Config.img_width, Config.img_channel])

    if is_train==True:
        image = image_augmentation(image)
    else:
        image = image
        image = tf.image.per_image_standardization(image)

    image = tf.cast(image, tf.float32)

    min_after_dequeue = Config.min_after_dequeue
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
                                             [image, label],
                                             batch_size=batch_size,
                                             num_threads= 64,
                                             capacity=capacity,
                                             min_after_dequeue=min_after_dequeue)
    else:
        # input_queue = tf.train.slice_input_producer([image, label], shuffle=False, num_epochs=1)
        image_batch, label_batch = tf.train.batch(
                                            [image, label],
                                            batch_size=batch_size,
                                            num_threads = 64,
                                            capacity=300
                                            )

    return image_batch, label_batch

def image_augmentation(image):
    flip_le_right = tf.image.random_flip_left_right(image)
    bright = tf.image.random_brightness(flip_le_right, max_delta=0.2)
    contrast = tf.image.random_contrast(bright, lower=0.1, upper=1.8)
    std_image = tf.image.per_image_standardization(contrast)
    return std_image

if __name__=="__main__":

    tfrecord_file = Config.test_tfrecord_file
    batch_size = 8
    image, label = read_tfrecord(tfrecord_file, batch_size, shuffle=False, is_train=False)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess, coord)
        for i in range(100):
            image_, label_ = sess.run([image,label])
            image0 = image_[0].reshape((224,224,3))
            cv2.namedWindow('i',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('i', image0)

            image1= image_[1].reshape((224,224,3))
            cv2.namedWindow('i1', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('i1', image1)
            cv2.waitKey(1000)






