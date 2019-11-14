img_width = 299
img_height = 299
img_channel = 3

batch_size = 32
# 对于按顺序制作的样本文件，这个值应该尽量涵盖样本数量，不然后面队列中的图片顺序不会被打乱
min_after_dequeue = 3500
classes={
         'dandelion':0,
         'daisy':1,
         'roses':2,
         'sunflowers':3,
         'tulips':4
         }
num_classes = 5
num_samples = 3414
test_num_samples = 250
batch_num = int(num_samples/batch_size)

learning_rate = 1e-2
epoch = 100

# train_dataset_dir = "F:/TF/flowersdata/train/"
# test_dataset_dir = "F:/TF/flowersdata/test/"
#
# tfrecord_file = "F:/TF/flowersdata/train/train.tfrecord"
# test_tfrecord_file = "F:/TF/flowersdata/test/test.tfrecord"
# train_log_dir = "F:/TF/AlexNet/Inceptionlogs/"

train_dataset_dir = "/home/g18661755180/zz/AlexNet/flowersdata/train/"
test_dataset_dir = "/home/g18661755180/zz/AlexNet/flowersdata/test/"

tfrecord_file ="/home/g18661755180/zz/AlexNet/train.tfrecord"
test_tfrecord_file ="/home/g18661755180/zz/AlexNet/test.tfrecord"
train_log_dir = "/home/g18661755180/zz/AlexNet/VGGlogs/"