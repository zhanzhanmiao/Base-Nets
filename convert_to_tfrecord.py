import os
import tensorflow as tf
import Config
import cv2

# 不同类的图片放在不同的文件夹
def create_tfrecord(filename, classes, train_data_dir):
    writer = tf.python_io.TFRecordWriter(filename)
    for name in classes:
        class_path = train_data_dir+str(name)+"/"
        for img_name in os.listdir(class_path):
            img_path = class_path+img_name
            # img = Image.open(img_path)
            img_ori = cv2.imread(img_path)
            img = cv2.resize(img_ori, (Config.img_width,Config.img_height), interpolation=cv2.INTER_CUBIC)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('input_image',img)
            #
            # cv2.namedWindow('input_image_ori', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('input_image_ori', img_ori)
            # cv2.waitKey()
            img_raw = img.tobytes()

            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[Config.classes[name]])),
                    "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
        print('finish ', name)
    writer.close()


if __name__=="__main__":
    train_dataset_dir = Config.train_dataset_dir
    tfrecord_filename = Config.tfrecord_file
    test_dataset_dir = Config.test_dataset_dir
    test_tfrecord_filename = Config.test_tfrecord_file

    classes = Config.classes.keys()
    create_tfrecord(tfrecord_filename, classes,train_dataset_dir)
    create_tfrecord(test_tfrecord_filename, classes, test_dataset_dir)