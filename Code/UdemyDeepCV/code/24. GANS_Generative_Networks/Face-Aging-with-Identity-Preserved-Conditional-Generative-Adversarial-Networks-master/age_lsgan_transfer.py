import os.path
import os
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import tensorflow as tf
from datetime import datetime
from models import FaceAging
import sys
sys.path.append('./tools/')
from source_input import load_source_batch3
from utils import save_images, save_source
from data_generator import ImageDataGenerator

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")

flags.DEFINE_integer("batch_size", 32, "The size of batch images")

flags.DEFINE_integer("image_size", 128, "the size of the generated image")

flags.DEFINE_integer("noise_dim", 256, "the length of the noise vector")

flags.DEFINE_integer("feature_size", 128, "image size after stride 2 conv")

flags.DEFINE_integer("age_groups", 5, "the number of different age groups")

flags.DEFINE_integer('max_steps', 200000, 'Number of batches to run')

flags.DEFINE_string("alexnet_pretrained_model", "pre_trained/alexnet.model-292000",
                    "Directory name to save the checkpoints")

flags.DEFINE_string("age_pretrained_model", "pre_trained/age_classifier.model-300000",
                    "Directory name to save the checkpoints")

flags.DEFINE_integer('model_index', None, 'the index of trained model')

flags.DEFINE_float("gan_loss_weight", None, "gan_loss_weight")

flags.DEFINE_float("fea_loss_weight", None, "fea_loss_weight")

flags.DEFINE_float("age_loss_weight", None, "age_loss_weight")

flags.DEFINE_float("tv_loss_weight", None, "face_loss_weight")

flags.DEFINE_string("checkpoint_dir", None, "Directory name to save the checkpoints")

flags.DEFINE_string("source_checkpoint_dir", ' ', "Directory name to save the checkpoints")

flags.DEFINE_string("sample_dir", None, "Directory name to save the sample images")

flags.DEFINE_string("fea_layer_name", None, "which layer to use for fea_loss")

flags.DEFINE_string("source_file", 'your training file', "source file path")

flags.DEFINE_string("root_folder", 'CACD_cropped_400/', "folder that contains images")

FLAGS = flags.FLAGS

# How often to run a batch through the validation model.
VAL_INTERVAL = 5000

# How often to save a model checkpoint
SAVE_INTERVAL = 10000

d_iter = 1
g_iter = 1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(batch_size=FLAGS.batch_size, height=FLAGS.feature_size, width=FLAGS.feature_size,
                                     z_dim=FLAGS.noise_dim, scale_size=(FLAGS.image_size, FLAGS.image_size), mode='train')
def my_train():
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        model = FaceAging(sess=sess, lr=FLAGS.learning_rate, keep_prob=1., model_num=FLAGS.model_index, batch_size=FLAGS.batch_size,
                        age_loss_weight=FLAGS.age_loss_weight, gan_loss_weight=FLAGS.gan_loss_weight,
                        fea_loss_weight=FLAGS.fea_loss_weight, tv_loss_weight=FLAGS.tv_loss_weight)

        imgs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
        true_label_features_128 = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, FLAGS.age_groups])
        true_label_features_64 = tf.placeholder(tf.float32, [FLAGS.batch_size, 64, 64, FLAGS.age_groups])
        false_label_features_64 = tf.placeholder(tf.float32, [FLAGS.batch_size, 64, 64, FLAGS.age_groups])
        age_label = tf.placeholder(tf.int32, [FLAGS.batch_size])

        source_img_227, source_img_128, face_label = load_source_batch3(FLAGS.source_file, FLAGS.root_folder, FLAGS.batch_size)

        model.train_age_lsgan_transfer(source_img_227, source_img_128, imgs, true_label_features_128,
                                       true_label_features_64, false_label_features_64, FLAGS.fea_layer_name, age_label)

        ge_samples = model.generate_images(imgs, true_label_features_128, reuse=True, mode='train')

        # Create a saver.
        model.saver = tf.train.Saver(model.save_d_vars + model.save_g_vars, max_to_keep=200)
        model.alexnet_saver = tf.train.Saver(model.alexnet_vars)
        model.age_saver = tf.train.Saver(model.age_vars)

        d_error = model.d_loss/model.gan_loss_weight
        g_error = model.g_loss/model.gan_loss_weight
        fea_error = model.fea_loss/model.fea_loss_weight
        age_error = model.age_loss/model.age_loss_weight

        # Start running operations on the Graph.
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess)

        model.alexnet_saver.restore(sess, FLAGS.alexnet_pretrained_model)
        model.age_saver.restore(sess, FLAGS.age_pretrained_model)

        if model.load(FLAGS.checkpoint_dir, model.saver):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print("{} Start training...")

        # Loop over max_steps
        for step in range(FLAGS.max_steps):
            images, t_label_features_128, t_label_features_64, f_label_features_64, age_labels = \
                train_generator.next_target_batch_transfer2()
            dict = {imgs: images,
                    true_label_features_128: t_label_features_128,
                    true_label_features_64: t_label_features_64,
                    false_label_features_64: f_label_features_64,
                    age_label: age_labels
                    }
            for i in range(d_iter):
                _, d_loss = sess.run([model.d_optim, d_error], feed_dict=dict)

            for i in range(g_iter):
                _, g_loss, fea_loss, age_loss = sess.run([model.g_optim, g_error, fea_error, age_error],
                                                         feed_dict=dict)

            format_str = ('%s: step %d, d_loss = %.3f, g_loss = %.3f, fea_loss=%.3f, age_loss=%.3f')
            print(format_str % (datetime.now(), step, d_loss, g_loss, fea_loss, age_loss))

            # Save the model checkpoint periodically.
            if step % SAVE_INTERVAL == SAVE_INTERVAL-1 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir)
                model.save(checkpoint_path, step, 'acgan')

            if step % VAL_INTERVAL == VAL_INTERVAL-1:
                if not os.path.exists(FLAGS.sample_dir):
                    os.makedirs(FLAGS.sample_dir)
                path = os.path.join(FLAGS.sample_dir, str(step))
                if not os.path.exists(path):
                    os.makedirs(path)

                source = sess.run(source_img_128)
                save_source(source, [4, 8], os.path.join(path, 'source.jpg'))
                for j in range(train_generator.n_classes):
                    true_label_fea = train_generator.label_features_128[j]
                    dict = {
                            imgs: source,
                            true_label_features_128: true_label_fea
                            }
                    samples = sess.run(ge_samples, feed_dict=dict)
                    save_images(samples, [4, 8], './{}/test_{:01d}.jpg'.format(path, j))

def main(argv=None):
    my_train()


if __name__ == '__main__':
    tf.app.run()
