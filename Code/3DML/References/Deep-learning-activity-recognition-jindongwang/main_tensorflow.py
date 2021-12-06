# encoding=utf-8
"""
    Created on 15:07 2017/8/16 
    @author: Jindong Wang
"""

import os

# This is for showing the Tensorflow log
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf


# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print(x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print(X.shape)
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(6)[data]
    return YY


# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data():
    import os
    if os.path.isfile('data/data_har.npz') == True:
        data = np.load('data/data_har.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
    else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        str_folder = 'Your root folder' + 'UCI HAR Dataset/'
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

    return X_train, Y_train, X_test, Y_test


# A class for some hyperparameters
class Config(object):
    def __init__(self, X_train, Y_train):
        self.n_input = len(X_train[0])  # number of input neurons to the network
        self.n_output = len(Y_train[0])  # number of output neurons
        self.dropout = 0.8  # dropout, between 0 and 1
        self.learning_rate = 0.001  # learning rate, float
        self.training_epoch = 20  # training epoch
        self.n_channel = 9  # number of input channel
        self.input_height = 128  # input height
        self.input_width = 1  # input width
        self.kernel_size = 64  # number of convolution kernel size
        self.depth = 32  # number of convolutions
        self.batch_size = 16  # batch size
        self.show_progress = 50  # how many batches to show the progress

        # weights and biases definition
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([1, self.kernel_size, self.n_channel, self.depth])),
            'wc2': tf.Variable(tf.random_normal([1, self.kernel_size, self.depth, 64])),
            'wd1': tf.Variable(tf.random_normal([32 * 32 * 2, 1000])),
            'wd2': tf.Variable(tf.random_normal([1000, 500])),
            'wd3': tf.Variable(tf.random_normal([500, 300])),
            'out': tf.Variable(tf.random_normal([300, self.n_output]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([self.depth])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1000])),
            'bd2': tf.Variable(tf.random_normal([500])),
            'bd3': tf.Variable(tf.random_normal([300])),
            'out': tf.Variable(tf.random_normal([self.n_output]))
        }


# wrap of conv1d
def conv1d(x, W, b, stride):
    x = tf.nn.conv2d(x, W, strides=[1, stride, 1, 1], padding='SAME')
    x = tf.add(x, b)
    return tf.nn.relu(x)


# wrap of maxpool1d
def maxpool1d(x, kernel_size, stride):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, 1, 1], strides=[1, stride, 1, 1], padding='VALID')


# network definition
def conv_net(x, W, b, dropout):
    conv1 = conv1d(x, W['wc1'], b['bc1'], 1)
    conv1 = maxpool1d(conv1, 2, stride=2)
    conv2 = conv1d(conv1, W['wc2'], b['bc2'], 1)
    conv2 = maxpool1d(conv2, 2, stride=2)
    conv2 = tf.reshape(conv2, [-1, W['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(conv2, W['wd1']), b['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout)
    fc2 = tf.add(tf.matmul(fc1, W['wd2']), b['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=dropout)
    fc3 = tf.add(tf.matmul(fc2, W['wd3']), b['bd3'])
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, keep_prob=dropout)
    out = tf.add(tf.matmul(fc3, W['out']), b['out'])
    return out


# wrap the network for training and testing
def network(X_train, Y_train, X_test, Y_test):
    config = Config(X_train, Y_train)

    # X, Y and keep_prob are three feeds to the network
    X = tf.placeholder(tf.float32, shape=[None, config.input_height, config.input_width, config.n_channel])
    Y = tf.placeholder(tf.float32, shape=[None, config.n_output])
    keep_prob = tf.placeholder(tf.float32)

    y_pred = conv_net(X, config.weights, config.biases, config.dropout)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.arg_max(y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    total_batch = len(X_train) // config.batch_size

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(config.training_epoch):
            for j in range(total_batch):
                x_train_batch, y_train_batch = X_train[j * config.batch_size: config.batch_size * (j + 1)], \
                                               Y_train[j * config.batch_size: config.batch_size * (j + 1)]
                x_train_batch = np.reshape(x_train_batch, [len(x_train_batch), 128, 1, 9])
                sess.run(optimizer, feed_dict={X: x_train_batch, Y: y_train_batch, keep_prob: config.dropout})
                if j % config.show_progress == 0:
                    loss, acc = sess.run([cost, accuracy],
                                         feed_dict={X: x_train_batch,
                                                    Y: y_train_batch,
                                                    keep_prob: config.dropout})
                    print('Epoch:%02d,batch:%03d,loss:%.8f,accuracy:%.8f' % (
                        i + 1, (j + 1) * config.batch_size, loss, acc))
        print('Optimization finished!')
        acc_test = sess.run(accuracy, feed_dict={X: np.reshape(X_test, [len(X_test), 128, 1, 9]),
                                                 Y: np.reshape(Y_test, [len(Y_test), 6]),
                                                 keep_prob: 1.})
        print('Accuracy of testing:%.8f' % acc_test)


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()
    # normalizing the data
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test = (X_test - np.mean(X_test)) / np.std(X_test)
    # build the network
    network(X_train, Y_train, X_test, Y_test)