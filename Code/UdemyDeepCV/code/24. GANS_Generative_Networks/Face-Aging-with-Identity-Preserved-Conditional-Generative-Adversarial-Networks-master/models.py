import tensorflow as tf
import numpy as np
import os.path
import random
from tensorflow.python.training import moving_averages
import sys
sys.path.append('./tools/')
from ops import *

class FaceAging(object):
    def __init__(self, sess, lr, keep_prob, model_num, batch_size=64, decay_steps=None,
                 gan_loss_weight=None, fea_loss_weight=None, age_loss_weight=None,
                 tv_loss_weight=None):

        self.sess = sess
        self.NUM_CLASSES = 2000
        self.KEEP_PROB = keep_prob
        self.train_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
        self.skip_layers = ['fc8']
        self.learning_rate = lr
        self.decay_steps = decay_steps
        self.learning_rate_decay_factor = 0.1
        self.model_num = model_num
        self._extra_train_ops = []
        self.batch_size = batch_size
        self.mean = tf.constant([104., 117., 124.])
        self.fea_loss_weight = fea_loss_weight
        self.age_loss_weight = age_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.tv_loss_weight = tv_loss_weight
        self.weight_decay_rate = 0.0005

    def inference(self, x, scope_name='alexnet', reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
            conv1 = conv(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
            norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

            # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
            conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
            pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
            norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

            # 3rd Layer: Conv (w ReLu)
            conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

            # 4th Layer: Conv (w ReLu) splitted into two groups
            conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

            # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
            conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

            # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
            flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
            fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
            dropout6 = dropout(fc6, self.KEEP_PROB)

            # 7th Layer: FC (w ReLu) -> Dropout
            fc7 = fc(dropout6, 4096, 4096, name='fc7')
            dropout7 = dropout(fc7, self.KEEP_PROB)
            # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
            self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')

            return scope

    def face_age_alexnet(self, x, scope_name='alexnet', if_age=False, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
            conv1 = conv(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
            norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

            # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
            conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
            pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
            norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

            # 3rd Layer: Conv (w ReLu)
            conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
            self.conv3 = conv3
            # 4th Layer: Conv (w ReLu) splitted into two groups
            conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
            self.conv4 = conv4

            # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
            conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
            self.conv5 = conv5

            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
            self.pool5 = pool5

            # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
            flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
            fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
            self.fc6 = fc6
            dropout6 = dropout(fc6, self.KEEP_PROB)

            # 7th Layer: FC (w ReLu) -> Dropout
            fc7 = fc(dropout6, 4096, 4096, name='fc7')
            self.fc7 = fc7
            dropout7 = dropout(fc7, self.KEEP_PROB)

            face_fc9 = fc(dropout7, 4096, 256, name='new_1')
            self.fc9 = face_fc9
            # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
            self.face_logits = fc(face_fc9, 256, self.NUM_CLASSES, relu=False, name='new_2')

            if if_age:
                age_fc6 = fc(flattened, 6 * 6 * 256, 4096, name='age_fc6')
                age_dropout6 = dropout(age_fc6, self.KEEP_PROB)
                # 7th Layer: FC (w ReLu) -> Dropout
                age_fc7 = fc(age_dropout6, 4096, 4096, name='age_fc7')
                age_dropout7 = dropout(age_fc7, self.KEEP_PROB)
                self.age_logits = fc(age_dropout7, 4096, 5, name='age_fc8', relu=False)

            return scope

    def ResnetGenerator(self, image, name, n_blocks=6, condition=None, mode='train', reuse=False):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if condition is not None:
                image = tf.concat([image, condition], axis=3)

            x = tf.nn.relu(self.batch_norm('bn1', conv2d(image, 32, 7, 7, d_h=1, d_w=1, name='conv1'), mode=mode))
            x = tf.nn.relu(self.batch_norm('bn2', conv2d(x, 64, 3, 3,  name='conv2'), mode=mode))

            # if condition is not None:
                # x = tf.concat([x, condition], axis=3)

            x = tf.nn.relu(self.batch_norm('bn3', conv2d(x, 128, 3, 3, name='conv3'), mode=mode))

            for i in range(n_blocks):
                with tf.variable_scope('unit_%d' % i):
                    x = self.residual(x, 3, 128)

            x_shape = x.get_shape().as_list()
            x = deconv2d(x, [x_shape[0], x_shape[1]*2, x_shape[1]*2, 64], 3, 3, name="deconv1")
            x = tf.nn.relu(self.batch_norm('bn4', x, mode=mode))

            x_shape = x.get_shape().as_list()
            x = deconv2d(x, [x_shape[0], x_shape[1]*2, x_shape[1]*2, 32], 3, 3, name="deconv2")
            x = tf.nn.relu(self.batch_norm('bn5', x, mode=mode))

            x = conv2d(x, 3, 7, 7, d_h=1, d_w=1, name='conv4')
            print(x.get_shape)

            return tf.nn.tanh(x)

    def PatchDiscriminator(self, image, name, condition=None, mode='train', reuse=False):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            x = lrelu(conv2d(image, 64, 4, 4, name='dis'))
            if condition is not None:
                x = tf.concat([x, condition], axis=3)

            x = lrelu(self.batch_norm('dis_bn0', conv2d(x, 128, 4, 4, name='dis_1'), mode=mode))

            x = lrelu(self.batch_norm('dis_bn1', conv2d(x, 256, 4, 4, name='dis_2'), mode=mode))

            x = lrelu(self.batch_norm('dis_bn2', conv2d(x, 512, 4, 4, d_h=1, d_w=1, name='dis_3'), mode=mode))

            x = conv2d(x, 512, 4, 4, d_h=1, d_w=1, name='dis_4')

            print(x.get_shape())

            return x

    # conditional lsgan + batchnorm
    def train_age_lsgan_transfer(self, source_img_227, source_img_128, imgs, true_label_fea_128, true_label_fea_64,
                                 false_label_fea_64, fea_layer_name, age_label):
        """
        :param source_img_227: remove mean and size is 227
        :param source_img_128: remove mean and size is 128
        :param imgs: range in [-1, 1]
        :param true_label_fea: the same size as imgs, has 5 channes
        :param false_label_fea: the same size as imgs, has 5 channels
        :return:
        """

        self.face_age_alexnet(source_img_227, if_age=True)
        if fea_layer_name == 'conv3':
            source_fea = self.conv3
        elif fea_layer_name == 'conv4':
            source_fea = self.conv4
        elif fea_layer_name == 'conv5':
            source_fea = self.conv5
        elif fea_layer_name == 'pool5':
            source_fea = self.pool5
        elif fea_layer_name == 'fc6':
            source_fea = self.fc6
        elif fea_layer_name == 'fc7':
            source_fea = self.fc7

        self.g_source = self.ResnetGenerator(source_img_128, name='generator', condition=true_label_fea_128)

        discriminator = self.PatchDiscriminator

        # real image, right label
        D1_logits = discriminator(imgs, name='discriminator', condition=true_label_fea_64)
        # real image, false label
        D2_logits = discriminator(imgs, name='discriminator', condition=false_label_fea_64, reuse=True)
        # fake image, true label
        D3_logits = discriminator(self.g_source, name='discriminator', condition=true_label_fea_64, reuse=True)

        d_loss_real = tf.reduce_mean(tf.square(D1_logits - 1.))
        d_loss_fake1 = tf.reduce_mean(tf.square(D2_logits))
        d_loss_fake2 = tf.reduce_mean(tf.square(D3_logits))

        self.d_loss = (1. / 2 * (d_loss_real + 1. / 2 * (d_loss_fake1 + d_loss_fake2))) * self.gan_loss_weight

        self.g_loss = (1. / 2 * tf.reduce_mean(tf.square(D3_logits - 1.))) * self.gan_loss_weight

        g_source = (self.g_source + 1.) * 127.5
        # self.tv_loss = total_variation_loss(g_source) * self.tv_loss_weight

        g_source = tf.image.resize_bilinear(images=g_source, size=[227, 227])
        g_source = g_source - self.mean

        self.face_age_alexnet(g_source, if_age=True, reuse=True)
        self.age_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                logits=self.age_logits, labels=age_label)) * self.age_loss_weight

        if fea_layer_name == 'conv3':
            ge_fea = self.conv3
        elif fea_layer_name == 'conv4':
            ge_fea = self.conv4
        elif fea_layer_name == 'conv5':
            ge_fea = self.conv5
        elif fea_layer_name == 'pool5':
            ge_fea = self.pool5
        elif fea_layer_name == 'fc6':
            ge_fea = self.fc6
        elif fea_layer_name == 'fc7':
            ge_fea = self.fc7

        self.fea_loss = self.fea_loss_weight * mse(ge_fea, source_fea)

        g_loss = self.g_loss + self.fea_loss + self.age_loss

        self.get_vars()

        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.d_loss,
                                                                                 var_list=self.d_vars)
        train_ops = [d_optim] + self._extra_train_ops
        self.d_optim = tf.group(*train_ops)

        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(g_loss, var_list=self.g_vars)
        train_ops = [g_optim] + self._extra_train_ops
        self.g_optim = tf.group(*train_ops)


    def generate_images(self, source_img_128, true_label_fea, stable_bn=False, reuse=False, mode='test'):

        g_source = self.ResnetGenerator(source_img_128, name='generator', condition=true_label_fea,
                                             mode=mode, reuse=reuse)
        if stable_bn:
            train_ops = self._extra_train_ops
            self.optim = tf.group(*train_ops)
        # self.get_vars()

        return g_source


    def residual(self, x, filter_width, out_channels, mode='train'):
        """Residual unit with 2 sub layers."""
        in_channels = x.get_shape().as_list()
        in_channels = in_channels[-1]
        orig_x = x

        with tf.variable_scope('sub1'):
            x = conv2d(x, out_channels, filter_width, filter_width, d_h=1, d_w=1,  name='conv1')
            x = self.batch_norm('bn1', x, mode=mode)
            x = tf.nn.relu(x)

        with tf.variable_scope('sub2'):
            x = conv2d(x, out_channels, filter_width, filter_width, d_h=1, d_w=1, name='conv2')
            x = self.batch_norm('bn2', x, mode=mode)


        with tf.variable_scope('sub_add'):
            if in_channels != out_channels:
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0],  [(out_channels - in_channels) // 2, (out_channels - in_channels) // 2]])
            x += orig_x

        return x

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def batch_norm(self, name, x, mode='train'):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                # tf.summary.histogram(mean.op.name, mean)
                # tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))

    def get_vars(self):

        t_vars = tf.global_variables()

        alexnet_vars = [var for var in t_vars if 'alexnet' in var.name]
        self.alexnet_vars = [var for var in alexnet_vars if 'age' not in var.name]
        self.age_vars = [var for var in t_vars if 'age' in var.name]

        self.save_d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.save_g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in self.save_g_vars:
            print(var.name)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

    def save(self, checkpoint_dir, step, prefix):
        model_name = prefix + '.model'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir, saver, prefix=None, model_num=None):
        print(" [*] Reading checkpoints...")
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if model_num and prefix:
            ckpt_name = prefix + '.model-' + str(model_num)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def load_model(self, model_name):
        print("load model" + model_name)
        self.saver.restore(self.sess, model_name)
