import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                            initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                             initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def conv1d(input, filter_width, out_channels, in_channels=None, stride=1,
           HeUniform=False, with_bias=True, name=None):
    with tf.variable_scope(name):
        if not in_channels:
            in_channels = input.get_shape()[-1]
        if HeUniform:
            n = filter_width * out_channels
            std = np.sqrt(2.0 / n)
        else:
            std=0.02

        kernel = tf.get_variable('weights', [filter_width, in_channels, out_channels],
                                 initializer=tf.random_normal_initializer(stddev=std))
        conv = tf.nn.conv1d(input, kernel, stride=stride, padding='SAME')

        if with_bias:
            biases = tf.get_variable('b', [out_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           padding='SAME', name="conv2d"):
    """
    computes a 2-D convolution given 4-D input and filter tensors
    Given an input tensor of shape [batch, in_height, in_width, in_channels]
    a filter tensor of shape [filter_height, filter_width,in_channels, out_channels]
    """
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    """
    input:  A 4-D Tensor of type float and shape [batch, height, width, in_channels].
    filter: A 4-D Tensor with the same type as value and shape [height, width, output_channels, in_channels]
    filter's in_channels dimension must match that of value.
    """
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, name, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()
    input_ = tf.reshape(input_, [shape[0], -1])
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        matrix = tf.get_variable("weights", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))

        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias


def unravel_argmax(argmax, shape):
    output_list = [argmax // (shape[2] * shape[3]),
                   argmax % (shape[2] * shape[3]) // shape[3]]
    return tf.pack(output_list)


def unpool_layer2x2_batch(bottom, argmax):
    bottom_shape = tf.shape(bottom)
    top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat(4, [t2, t3, t1])
    indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])

    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))

    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

def UnPooling2x2ZeroFilled(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat(3, [x, tf.zeros_like(x)])
    out = tf.concat(2, [out, tf.zeros_like(out)])

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
        ret.set_shape([None, None, None, sh[3]])
        return ret

def FixedUnPooling(x, shape, unpool_mat=None):
    """
    Unpool the input with a fixed matrix to perform kronecker product with.
    Args:
        x (tf.Tensor): a NHWC tensor
        shape: int or (h, w) tuple
        unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
            If is None, will use a matrix with 1 at top-left corner.
    Returns:
        tf.Tensor: a NHWC tensor.
    """
    # shape = shape2d(shape)

    # a faster implementation for this special case
    if shape[0] == 2 and shape[1] == 2 and unpool_mat is None:
        return UnPooling2x2ZeroFilled(x)

    input_shape = x.get_shape().as_list()
    if unpool_mat is None:
        mat = np.zeros(shape, dtype='float32')
        mat[0][0] = 1
        unpool_mat = tf.constant(mat, name='unpool_mat')
    elif isinstance(unpool_mat, np.ndarray):
        unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
    assert unpool_mat.get_shape().as_list() == list(shape)

    # perform a tensor-matrix kronecker product
    fx = flatten(tf.transpose(x, [0, 3, 1, 2]))
    fx = tf.expand_dims(fx, -1)       # (bchw)x1
    mat = tf.expand_dims(flatten(unpool_mat), 0)  # 1x(shxsw)
    prod = tf.matmul(fx, mat)  # (bchw) x(shxsw)
    prod = tf.reshape(prod, [-1, input_shape[3], input_shape[1], input_shape[2], shape[0], shape[1] ])
    prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod = tf.reshape(prod, [-1, input_shape[1] * shape[0], input_shape[2] * shape[1], input_shape[3] ])

    return prod

def flatten(x):
    """
    Flatten the tensor.
    """
    return tf.reshape(x, [-1])


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")

        return tf.reduce_mean(-(targets * tf.log(preds + eps) + (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def mse(pred, target):
    num = pred.get_shape().as_list()[0]
    pred = tf.reshape(pred, [num, -1])
    target = tf.reshape(target, [num, -1])
    mse_sum = tf.reduce_sum(tf.pow(tf.subtract(pred, target), 2.0), 1)
    mse_loss = tf.reduce_mean(mse_sum)

    return mse_loss


def instance_norm(x, name):
    with tf.variable_scope(name):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

    return out


def l1_loss(pred, target):
    return tf.reduce_mean(tf.abs(pred - target))


def l2_loss(pred, target):
    return 1. / 2 * tf.reduce_mean(tf.pow(tf.subtract(pred, target), 2.0))


def tv_loss(images):
    return tf.reduce_mean(tf.image.total_variation(images))

def mmd_loss(source, target):

    source_mean = tf.reduce_mean(source)
    target_mean = tf.reduce_mean(target)

    mse_sum = tf.reduce_sum(tf.pow(tf.subtract(source_mean, target_mean), 2.0))*0.5

    return mse_sum