import os.path
import os
#os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import numpy as np
import tensorflow as tf
import time
from models import FaceAging
import sys
sys.path.append('./tools/')
from utils import save_images, save_source
from data_generator import ImageDataGenerator

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")

flags.DEFINE_integer("batch_size", 32, "The size of batch images")

flags.DEFINE_integer("image_size", 128, "the size of the generated image")

flags.DEFINE_integer("noise_dim", 256, "the length of the noise vector")

flags.DEFINE_integer("feature_size", 128, "image size after stride 2 conv")

flags.DEFINE_integer("age_groups", 5, "the number of different age groups")

flags.DEFINE_integer('model_index', None, 'the index of trained model')

flags.DEFINE_float("gan_loss_weight", None, "gan_loss_weight")

flags.DEFINE_float("fea_loss_weight", None, "fea_loss_weight")

flags.DEFINE_float("age_loss_weight", None, "age_loss_weight")

flags.DEFINE_float("tv_loss_weight", None, "face_loss_weight")

flags.DEFINE_string("checkpoint_dir", './checkpoints/0_conv5_lsgan_transfer_g75_0.5f-4_a30/',
                    "Directory name to save the checkpoints")

flags.DEFINE_string("save_dir", 'age/0_conv5_lsgan_transfer_g75_0.5f-4_a30/',
                    "Directory name to save the sample images")

flags.DEFINE_string("test_data_dir", './images/test/', "test images")

flags.DEFINE_string("train_data_dir", './images/train/', "train images")

FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

generator = ImageDataGenerator(batch_size=FLAGS.batch_size, height=FLAGS.feature_size, width=FLAGS.feature_size,
                               z_dim=FLAGS.noise_dim, scale_size=(FLAGS.image_size, FLAGS.image_size),
                               shuffle=False, mode='train')

val_generator = ImageDataGenerator(batch_size=FLAGS.batch_size, height=FLAGS.feature_size, width=FLAGS.feature_size,
                                   z_dim=FLAGS.noise_dim, scale_size=(FLAGS.image_size, FLAGS.image_size),
                                   shuffle=False, mode='test')


def my_train():
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        model = FaceAging(sess=sess, lr=FLAGS.learning_rate, keep_prob=1., model_num=FLAGS.model_index, batch_size=FLAGS.batch_size,
                        age_loss_weight=FLAGS.age_loss_weight, gan_loss_weight=FLAGS.gan_loss_weight,
                        fea_loss_weight=FLAGS.fea_loss_weight, tv_loss_weight=FLAGS.tv_loss_weight)

        model.imgs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
        model.true_label_features_128 = tf.placeholder(tf.float32, [FLAGS.batch_size, 128, 128, FLAGS.age_groups])

        model.ge_samples = model.generate_images(model.imgs, model.true_label_features_128, stable_bn=False, mode='train')

        model.get_vars()

        # Create a saver.
        model.saver = tf.train.Saver(model.save_g_vars)

        # Start running operations on the Graph.
        sess.run(tf.global_variables_initializer())

        if model.load(FLAGS.checkpoint_dir, model.saver, 'acgan', 399999):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print("{} Start testing...")
        if not os.path.exists(FLAGS.save_dir):
            os.makedirs(FLAGS.save_dir)

        # stable_bn(model, sess, 10000)
        # generate_images(model, sess)
        generate_images_from_folder(model, sess, FLAGS.test_data_dir, FLAGS.train_data_dir)

def generate_images_from_folder(model, sess, test_data_dir="./images/test", train_data_dir="./images/train"):
    if test_data_dir:
        source, paths = val_generator.load_imgs(test_data_dir, 128)
    else:
        source, paths = val_generator.next_source_imgs(0, 128, batch_size=256)

    if train_data_dir:
        train_imgs, _ = generator.load_imgs(train_data_dir, 128)
    else:
        train_imgs, _ = generator.next_source_imgs(0, 128, batch_size=FLAGS.batch_size-1)

    assert train_imgs.shape[0] == (FLAGS.batch_size-1)

    for i in range(len(paths)):
        print (i)
        temp = np.reshape(source[i], (1, 128, 128, 3))
        save_source(temp, [1, 1], os.path.join(FLAGS.save_dir, paths[i]))
        images = np.concatenate((temp, train_imgs), axis=0)
        for j in range(1, generator.n_classes):
            true_label_fea = generator.label_features_128[j]
            dict = {
                    model.imgs: images,
                    model.true_label_features_128: true_label_fea,
                    }
            samples = sess.run(model.ge_samples, feed_dict=dict)
            image = np.reshape(samples[0, :, :, :], (1, 128, 128, 3))
            # generator.save_batch(samples, paths, FLAGS.save_dir, index=j, if_target=True)
            save_images(image, [1, 1], os.path.join(FLAGS.save_dir, paths[i] + '_' + str(j) + '.jpg'))



def generate_images(model, sess):
    source, paths = val_generator.next_source_imgs(0, 128, batch_size=FLAGS.batch_size)
    time1 = time.time()
    for j in range(1, generator.n_classes):
        true_label_fea = generator.label_features_128[j]
        dict = {
                model.imgs: source,
                model.true_label_features_128: true_label_fea,
                }
        samples = sess.run(model.ge_samples, feed_dict=dict)
        # image = np.reshape(samples[0, :, :, :], (1, 128, 128, 3))
        save_images(samples, [1, 1], os.path.join(FLAGS.save_dir, paths[0] + '_' + str(j) + '.jpg'))

    time2 = time.time() - time1
    print (time2)



def stable_bn(model, sess, num_iter):
    for iter in range(num_iter):
        print (iter)
        train_imgs, _ = generator.next_source_imgs(0, 128, batch_size=FLAGS.batch_size)
        for j in range(1, generator.n_classes):
            true_label_fea = generator.label_features_128[j]
            dict = {
                model.imgs: train_imgs,
                model.true_label_features_128: true_label_fea,
            }
            sess.run([model.ge_samples, model.optim], feed_dict=dict)

    checkpoint_path = os.path.join(FLAGS.checkpoint_dir)
    model.save(checkpoint_path, 1000000, 'acgan')


def main(argv=None):
    my_train()


if __name__ == '__main__':
    tf.app.run()
