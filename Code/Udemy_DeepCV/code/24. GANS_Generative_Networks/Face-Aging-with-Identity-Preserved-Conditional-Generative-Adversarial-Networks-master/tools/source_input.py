
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

#from read_image import *
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
import numpy as np
import scipy.io as scio
from tensorflow.python.framework import ops
from PIL import Image
FLAGS = flags.FLAGS

T = 1
IM_HEIGHT = 400
IM_WIDTH = 400
IM_CHANNELS = 3


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_images(filename_queue, new_height=None, new_width=None):

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)  # use png or jpeg decoder based on your files
    image = tf.reshape(image, [IM_HEIGHT, IM_WIDTH, IM_CHANNELS])

    if new_height and new_width:
        image = tf.image.resize_images(image, [new_height, new_width])

    image = tf.cast(image, tf.float32) - np.array([104., 117., 124.])

    return image


def read_images2(filename_queue):

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)  # use png or jpeg decoder based on your files
    image = tf.reshape(image, [IM_HEIGHT, IM_WIDTH, IM_CHANNELS])

    image_227 = tf.image.resize_images(image, [227, 227])
    image_227 = tf.cast(image_227, tf.float32) - np.array([104., 117., 124.])

    image_128 = tf.image.resize_images(image, [128, 128])
    image_128 = tf.cast(image_128, tf.float32) - np.array([104., 117., 124.])

    return image_227, image_128

def read_images3(input_queue):

    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_image(file_contents, channels=3)
    image = tf.reshape(image, [IM_HEIGHT, IM_WIDTH, IM_CHANNELS])

    image_227 = tf.image.resize_images(image, [227, 227])
    image_227 = tf.cast(image_227, tf.float32) - np.array([104., 117., 124.])

    image_128 = tf.image.resize_images(image, [128, 128])
    # image_128 = tf.cast(image_128, tf.float32)
    image_128 = tf.cast(image_128, tf.float32) - np.array([104., 117., 124.])

    return image_227, image_128, label

def load_source_batch(filename, img_folder, batch_size, img_size, shuffle=True):
    filenames = get_imgAndlabel_list(filename, img_folder)
    print('%d images to train' %(len(filenames)))
    if not filenames:
        raise RuntimeError('No data files found.')

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename queue.
        image = read_images(filename_queue, new_height=img_size, new_width=img_size)
        image_batch = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=4,
            capacity=1280,
            min_after_dequeue=640)
        # image_batch = tf.train.batch(
        #     [image],
        #     batch_size=batch_size,
        #     num_threads=4,
        #     capacity=1280)
        #
        return image_batch

def load_source_batch2(filename, img_folder, batch_size, shuffle=True):

    filenames = get_imgAndlabel_list(filename, img_folder)
    print('%d images to train' % (len(filenames)))
    if not filenames:
        raise RuntimeError('No data files found.')

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename queue.
        image_227, image_128 = read_images2(filename_queue)
        image_227_batch, image_128_batch = tf.train.shuffle_batch(
            [image_227, image_128],
            batch_size=batch_size,
            num_threads=4,
            capacity=1280,
            min_after_dequeue=640)

        return image_227_batch, image_128_batch

def load_source_batch3(filename, img_folder, batch_size, shuffle=True):

    img_list, label_list = get_imgAndlabel_list2(filename, img_folder)
    print('%d images to train' % (len(img_list)))

    images = ops.convert_to_tensor(img_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=shuffle)

    # Even when reading in multiple threads, share the filename queue.
    image_227, image_128, label = read_images3(input_queue)
    image_227_batch, image_128_batch, label_batch = tf.train.shuffle_batch(
        [image_227, image_128, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=1280,
        min_after_dequeue=640)

    return image_227_batch, image_128_batch, label_batch

def get_imgAndlabel_list(filename, img_folder):
    """
    :param filename:
     each line in filename is img_name \space label
    :return:
    img names list
    label list
    """
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    imgname_lists = []
    for i in range(len(lines)):
        img_name = lines[i].split()[0]
        imgname_lists.append(os.path.join(img_folder, img_name))
    return imgname_lists

def get_imgAndlabel_list2(filename, img_folder):
    """
    :param filename:
     each line in filename is img_name \space label
    :return:
    img names list
    label list
    """
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    imgname_lists = []
    label_lists = []
    for i in range(len(lines)):
        img_name, label = lines[i].split()
        imgname_lists.append(os.path.join(img_folder, img_name))
        label_lists.append(int(label))

    return imgname_lists, label_lists
