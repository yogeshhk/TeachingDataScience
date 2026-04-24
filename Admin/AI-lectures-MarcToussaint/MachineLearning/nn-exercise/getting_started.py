import tensorflow as tf
import numpy as np


sess = tf.Session()

X = tf.placeholder(tf.float32, shape=(100, 3))
y = tf.placeholder(tf.float32, shape=(100))
beta = tf.placeholder(tf.float32, shape=(3))

p = tf.math.sigmoid(tf.tensordot(X, beta, 1))
Loss = -tf.math.reduce_sum(y * tf.math.log(p) + ((1. - y) * tf.math.log(1.-p)))

dL = tf.gradients(Loss, beta)
ddL = tf.hessians(Loss, beta)

rand_X = np.random.uniform(-1, 1, (100,3))
rand_y = np.random.randint(0, 2, 100)
rand_beta = np.random.uniform(-1, 1, 3)

print(sess.run([Loss, dL, ddL], feed_dict={beta: rand_beta, X: rand_X, y: rand_y}))

writer = tf.summary.FileWriter('logs', sess.graph)
writer.close()


def numpy_equations(X, beta, y):
    p = 1. / (1. + np.exp(-np.dot(X, beta)))
    L = -np.sum(y * np.log(p) + ((1. - y) * np.log(1.-p)))
    dL = np.dot(X.T, p - y)
    W = np.identity(X.shape[0]) * p * (1. - p)
    ddL = np.dot(X.T, np.dot(W, X))
    return L, dL, ddL

print(numpy_equations(rand_X, rand_beta, rand_y))
