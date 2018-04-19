import os
import tensorflow as tf
from PIL import Image
import random

cwd = os.getcwd()
file_dir = 'F:\\001-python\\Data\\catdog\\test\\'
recordpath="F:\\001-python\\train120.tfrecords"
filelist = []


def create_record_list():
    for file in os.listdir(file_dir):
        filelist.append(file)
        '''
        name = file.split(sep='.')
        lable_val = 0
        if name[0] == 'cat':
            lable_val = 0
        else:
            lable_val = 1
        img_path = file_dir + file
        img = Image.open(img_path)
        img = img.resize((208, 208))
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lable_val])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))

        writer.write(example.SerializeToString())
        i=i+1
        print(i)
        '''

# 生成是数据文件
def create_record(filelist):
    random.shuffle(filelist)
    i = 0
    writer = tf.python_io.TFRecordWriter(recordpath)
    for file in filelist:
        name = file.split(sep='.')
        lable_val = 0
        if name[0] == 'cat':
            lable_val = 0
        else:
            lable_val = 1
        img_path = file_dir + file
        img = Image.open(img_path)
        img = img.resize((240, 240))
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lable_val])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #example对象对label和image进行封装
        writer.write(example.SerializeToString())
        i=i+1
        print(name[1])
        print(lable_val)
        print(i)
    writer.close()

# 用队列形式读取文件
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

# img, label = read_and_decode("F:\\001-python\\train.tfrecords")
#
# # 使用shuffle_batch可以随机打乱输入
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                 batch_size=50, capacity=2000,
#                                                 min_after_dequeue=1000)
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     threads = tf.train.start_queue_runners(sess=sess)
#     for i in range(3):
#         val, L = sess.run([img_batch, label_batch])
#
#         # 我们也可以根据需要对val， l进行处理
#         # l = to_categorical(l, 12)
#         print(val, L)

# if __name__ == '__main__':
#     create_record_list()
#     create_record(filelist)
