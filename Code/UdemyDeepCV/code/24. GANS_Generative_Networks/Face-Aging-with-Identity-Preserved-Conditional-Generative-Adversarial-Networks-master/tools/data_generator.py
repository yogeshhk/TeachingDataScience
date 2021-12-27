import numpy as np
import cv2
import random
from PIL import Image
import os.path
"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
"""


class ImageDataGenerator:
    def __init__(self, batch_size, height, width, z_dim, shuffle=True,
                 scale_size=(64, 64), classes=5, mode='train'):

        # Init params
        self.root_folder = '/new_disk2/wangzw/tangxu/CACD_cropped_400/'
        if mode == 'train':
            self.file_folder = '/new_disk2/wangzw/tangxu/age_data/train_data/'
            self.class_lists = ['train_age_group_0.txt',
                               'train_age_group_1.txt',
                               'train_age_group_2.txt',
                               'train_age_group_3.txt',
                               'train_age_group_4.txt']
            self.pointer = [0, 0, 0, 0, 0]
        else:
            self.file_folder = '/new_disk2/wangzw/tangxu/age_data/test_data/'
            self.class_lists = ['test_age_group_0.txt',
                               'test_age_group_1.txt',
                               'test_age_group_2.txt',
                               'test_age_group_3.txt',
                               'test_age_group_4.txt']
            self.pointer = [0, 0, 0, 0, 0, 0]

        self.train_label_pair = '/home/wangzw/Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks-master/tools' \
                                '/train_label_pair.txt'
        self.true_labels = []
        self.false_labels = []
        self.images = []
        self.labels = []
        self.data_size = []
        self.n_classes = classes
        self.shuffle = shuffle
        self.scale_size = scale_size
        self.label_pair_index = 0

        self.mean = np.array([104., 117., 124.])
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.z_dim = z_dim
        self.img_size = self.height

        #self.read_class_list(self.class_lists)
        if self.shuffle:
            self.shuffle_data(shuffle_all=True)

        self.get_age_labels()
        self.label_features_128, _ = self.pre_generate_labels(batch_size, 128, 128)
        self.label_features_64, self.one_hot_labels = self.pre_generate_labels(batch_size, 64, 64)

    def __iter__(self):
        return self

    def get_age_labels(self):
        batch_size = self.batch_size
        self.age_label = []
        self.age_label.append(np.zeros(batch_size, np.int32))
        self.age_label.append(np.ones(batch_size, np.int32))
        self.age_label.append(np.ones(batch_size, np.int32)*2)
        self.age_label.append(np.ones(batch_size, np.int32) * 3)
        self.age_label.append(np.ones(batch_size, np.int32) * 4)

    def pre_generate_labels(self, batch_size, height, width):
        # features and one hot labels for each sample, n_class kinds
        features = []
        one_hot_labels = []
        full_1 = np.ones((height, width))

        for i in range(self.n_classes):
            temp = np.zeros((height, width, self.n_classes))
            temp[:, :, i] = full_1
            features.append(temp)

            temp = np.zeros((1, self.n_classes))
            temp[0, i] = 1
            one_hot_labels.append(temp)

        # features and one hot labels for a batch, n_class kinds
        batch_label_features = []
        batch_one_hot_labels = []
        for i in range(self.n_classes):
            temp_label_features = np.zeros((batch_size, height, width, self.n_classes))
            temp_label = np.zeros((batch_size, self.n_classes))

            for j in range(batch_size):
                temp_label_features[j, :, :, :] = features[i]
                temp_label[j, :] = one_hot_labels[i]

            batch_label_features.append(temp_label_features)
            batch_one_hot_labels.append(temp_label)

        return batch_label_features, batch_one_hot_labels

    def read_class_list(self, class_lists):
        """
        Scan the image file and get the image paths and labels
        """
        for i in range(len(class_lists)):
            f = open(self.file_folder + class_lists[i], 'r')
            lines = f.readlines()
            f.close()
            images = []
            labels = []
            for l in lines:
                items = l.split()
                images.append(items[0])
                labels.append(int(i))

            self.images.append(images)
            self.labels.append(labels)
            # store total number of data
            self.data_size.append(len(labels))

        with open(self.train_label_pair) as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                item = line.split()
                self.true_labels.append(int(item[0]))
                self.false_labels.append(int(item[1]))

    def next_batch(self):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory
        """
        # Get next batch of image (path) and labels
        index = random.randint(0, 4)

        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]

        # Read images
        images = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            images[i] = process_target_img(self.root_folder, paths[i], self.scale_size[0])

        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)

        label_list = [0, 1, 2, 3, 4]
        label_list.remove(index)
        random.shuffle(label_list)
        error_label = label_list[0]

        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        return images, batch_z, self.one_hot_labels[index], self.label_features_64[index], \
               self.label_features_64[error_label], index

    def mp_next_batch(self):
        index = random.randint(0, 4)
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)
        pool = mp.Pool(processes=4)
        images = [pool.apply_async(process_target_img, args=(self.root_folder, path, self.scale_size[0]))
                  for path in paths]
        images = [p.get() for p in images]
        images = np.concatenate(images, axis=0)

        label_list = [0, 1, 2, 3, 4]
        label_list.remove(index)
        random.shuffle(label_list)
        error_label = label_list[0]

        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        return images, batch_z, self.one_hot_labels[index], self.label_features_64[index], \
               self.label_features_64[error_label], index



    def shuffle_data(self, index=None, shuffle_all=False):
        """
        Random shuffle the images, since one group images has the same label, so we do not shuffle labels
        """
        if shuffle_all:
            for i in range(len(self.images)):
                random.shuffle(self.images[i])
        else:
            if index:
                random.shuffle(self.images[index])

    def reset_pointer(self, index):
        """
        reset pointer to begin of the list
        """
        self.pointer[index] = 0

        if self.shuffle:
            self.shuffle_data(index)

    def process_source_img(self, img_path, image_size, mean, scale):
        img = cv2.imread(self.root_folder + img_path)
        img = img[:, :, [2, 1, 0]]
        # rescale image
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype(np.float32)
        img = (img - mean) * scale
        return img

    def next_source_imgs(self, index, image_size, batch_size, mean=np.array([104., 117., 124.]), scale=1.):
        paths = self.images[index][self.pointer[index]:self.pointer[index] + batch_size]
        # Read images
        images = np.ndarray([batch_size, image_size, image_size, 3])
        for i in range(len(paths)):
            images[i] = self.process_source_img(paths[i], image_size, mean, scale)

        # update pointer
        self.pointer[index] += batch_size
        if self.pointer[index] >= (self.data_size[index] - batch_size):
            self.reset_pointer(index)

        return images, paths

    def next_batch_transfer(self, source_index, source_image_size=227):
        index = self.true_labels[self.label_pair_index]
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        imgs = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            imgs[i] = process_target_img(self.root_folder, paths[i], self.scale_size[0])
        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)

        error_label = self.false_labels[self.label_pair_index]
        self.label_pair_index += 1

        source_imgs = self.next_source_imgs(source_index, source_image_size)

        return imgs, source_imgs, self.one_hot_labels[index], self.label_features[index], \
               self.label_features[error_label], index

    def next_age_batch_transfer(self, source_index):
        index = self.true_labels[self.label_pair_index]
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        imgs = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            imgs[i] = process_target_img(self.root_folder, paths[i], self.scale_size[0])
        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)

        error_label = self.false_labels[self.label_pair_index]
        self.label_pair_index += 1

        source_imgs = self.next_source_imgs(source_index)

        return imgs, source_imgs, self.one_hot_labels[index], self.label_features[index], \
               self.label_features[error_label], self.age_label[index], index

    def next_target_imgs(self, index):
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        imgs = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            imgs[i] = process_target_img(self.root_folder, paths[i], self.scale_size[0])
        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index]- self.batch_size):
            self.reset_pointer(index)

        return imgs

    def next_gan_batch(self):
        index = random.randint(0, 4)
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        imgs = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            imgs[i] = process_target_img2(self.root_folder, paths[i], self.scale_size[0])
        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index]- self.batch_size):
            self.reset_pointer(index)

        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        return imgs, batch_z

    def load_batch(self, index):
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        imgs = np.ndarray([self.batch_size, 227, 227, 3])
        for i in range(len(paths)):
            imgs[i] = process_source_img(self.root_folder, paths[i], self.mean)
        # update pointer
        self.pointer[index] += self.batch_size

        return imgs, self.one_hot_labels[index], paths

    def load_imgs(self, data_dir, img_size=128):
        paths = os.listdir(data_dir)
        # Read images
        imgs = np.ndarray([len(paths), img_size, img_size, 3])
        for i in range(len(paths)):
            img = cv2.imread(os.path.join(data_dir, paths[i]))
            img = img[:, :, [2, 1, 0]]
            # rescale image
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32)
            img -= self.mean
            imgs[i] = img

        return imgs, paths

    def save_batch(self, batch_imgs, img_names, folder, index=None, if_target=True):
        assert batch_imgs.shape[0] == len(img_names), 'img nums must match img names'
        shape = batch_imgs.shape[1:]
        for i in range(batch_imgs.shape[0]):
            img = np.reshape(batch_imgs[i, :, :, :], shape)
            if if_target:
                im = np.uint8((img + 1.)*127.5)
            else:
                im = np.uint8((img + self.mean))

            if (im.shape[2] == 1):
                im = Image.fromarray(np.reshape(im, [im.shape[0], im.shape[1]]), 'L')  # gray image
            else:
                im = Image.fromarray(im)
            if index is not None:
                im.save(os.path.join(folder, img_names[i] + '_' + str(index) + '.jpg'))
            else:
                im.save(os.path.join(folder, img_names[i]))

    def next_source_imgs2(self, index):
        # index = 0
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        images_227 = np.ndarray([self.batch_size, 227, 227, 3])
        images_128 = np.ndarray([self.batch_size, 128, 128, 3])
        for i in range(len(paths)):
            image = cv2.imread(self.root_folder + paths[i])
            image = image[:, :, [2, 1, 0]]
            # rescale image
            img = cv2.resize(image, (227, 227))
            img = img.astype(np.float32)
            img -= self.mean
            images_227[i] = img

            img = cv2.resize(image, (128, 128))
            img = img.astype(np.float32)
            img -= self.mean
            images_128[i] = img
        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index]- self.batch_size):
            self.reset_pointer(index)

        return images_227, images_128

    def next_batch_transfer2(self, source_index=0):
        index = self.true_labels[self.label_pair_index]
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        imgs = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            imgs[i] = process_target_img(self.root_folder, paths[i], self.scale_size[0])
        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)

        error_label = self.false_labels[self.label_pair_index]
        self.label_pair_index += 1

        images_227, images_128 = self.next_source_imgs2(source_index)

        return imgs, images_227, images_128, self.one_hot_labels[index], self.label_features[index], \
               self.label_features[error_label], self.age_label[index], index

    def next_target_batch_transfer(self):
        index = self.true_labels[self.label_pair_index]
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        imgs = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            imgs[i] = process_target_img(self.root_folder, paths[i], self.scale_size[0])
        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)

        error_label = self.false_labels[self.label_pair_index]
        self.label_pair_index += 1

        return imgs, self.one_hot_labels[index], self.label_features_64[index], \
               self.label_features_64[error_label], self.age_label[index]

    def next(self):
        index = self.true_labels[self.label_pair_index]
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        # Read images
        imgs = np.ndarray([self.batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            imgs[i] = process_target_img(self.root_folder, paths[i], self.scale_size[0])

        # update pointer
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)

        error_label = self.false_labels[self.label_pair_index]
        self.label_pair_index += 1

        return (imgs, self.label_features_128[index], self.label_features_64[index],
                self.label_features_64[error_label], self.age_label[index])

    def my_next_batch(self):
        index = random.randint(0, 4)
        paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
        self.pointer[index] += self.batch_size
        if self.pointer[index] >= (self.data_size[index] - self.batch_size):
            self.reset_pointer(index)

        folder_lists = [self.root_folder for i in range(self.batch_size)]
        img_sizes = [self.scale_size[0] for i in range(self.batch_size)]
        imgs = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for img in executor.map(process_target_img, folder_lists, paths, img_sizes):
                imgs.append(img)
        imgs = np.array(imgs)

        label_list = [0, 1, 2, 3, 4]
        label_list.remove(index)
        random.shuffle(label_list)
        error_label = label_list[0]
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        return imgs, batch_z, self.one_hot_labels[index], self.label_features[index], self.label_features[error_label], index


def process_target_img(root_folder, img_path, img_size):
    img = cv2.imread(root_folder + img_path)
    img = img[:, :, [2, 1, 0]]
    # rescale image
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    img = img / 127.5 - 1.
    return img


def process_target_img2(root_folder, img_path, img_size):
    img = cv2.imread(root_folder + img_path)
    img = img[:, :, [2, 1, 0]]
    # rescale image
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    img = img / 255.
    return img


def process_source_img(root_folder, img_path, mean):
    img = cv2.imread(root_folder + img_path)
    img = img[:, :, [2, 1, 0]]
    # rescale image
    img = cv2.resize(img, (227, 227))
    img = img.astype(np.float32)
    img -= mean
    return img


def load_target_batch(self):
    index = random.randint(1, 4)
    paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
    # update pointer
    self.pointer[index] += self.batch_size
    if self.pointer[index] >= (self.data_size[index] - self.batch_size):
        self.reset_pointer(index)
    # Read images
    folder_lists = [self.root_folder for i in range(self.batch_size)]
    img_sizes = [self.scale_size[0] for i in range(self.batch_size)]
    imgs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for img in executor.map(process_target_img, folder_lists, paths, img_sizes):
            imgs.append(img)
    imgs = np.array(imgs)

    return imgs


def next_source_imgs(self):
    index = 1
    paths = self.images[index][self.pointer[index]:self.pointer[index] + self.batch_size]
    # update pointer
    self.pointer[index] += self.batch_size
    if self.pointer[index] >= (self.data_size[index] - self.batch_size):
        self.reset_pointer(index)
    # Read images
    folder_lists = [self.root_folder for i in range(self.batch_size)]
    img_means = [self.mean for i in range(self.batch_size)]
    imgs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for img in executor.map(process_source_img, folder_lists, paths, img_means):
            imgs.append(img)
    return imgs

# if __name__ == '__main__':
#     generator = ImageDataGenerator(32, 128, 128, 256, shuffle=True,
#                      scale_size=(227, 227), classes=5, mode='train')
#     time1 = time.time()
#     generator.next_batch()
#     print(time.time() - time1)
#
#     time1 = time.time()
#     # for i in range(10):
#     generator.mp_next_batch()
#     print(time.time() - time1)
