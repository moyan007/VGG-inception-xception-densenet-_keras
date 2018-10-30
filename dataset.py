import os
import csv
import random

import numpy as np

from glob import glob
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
#from keras.applications.inception_v3 import preprocess_input

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit

DATA_DIR = './data/'
TARGET_TRAIN = DATA_DIR + 'train/'+ 'train.csv'
TARGET_TRAIN_DIR = DATA_DIR + 'train/data/'
TARGET_TEST = DATA_DIR + 'test.csv'
TARGET_TEST_DIR = DATA_DIR + 'test/'
#获取数据
class Dataset(object):
    def __init__(self, batch_size, image_shape):
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.train = self.get_data('train')
        self.val = self.get_data('val')
        self.test = self.get_data('test')
        self.classes_train = self.get_classes(self.train)
        self.classes_val = self.get_classes(self.val)

    def get_data(self, train_or_test):
        if (train_or_test == 'train' or train_or_test == 'val'):
            data_file = TARGET_TRAIN
            with open(data_file, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)[1:]
                temp_A = []
                temp_B = []
                for a, b in data:
                    temp_A.append(a)
                    temp_B.append(b)
            x_train, y_train, x_test, y_test = train_test_split(temp_A, temp_B,
                                                                    test_size=0.2, random_state=None,stratify=temp_B)
            data0 = []
            data1 = []
            for i in range(len(x_train)):
                data0.append([x_train[i],x_test[i]])
            for j in range(len(y_train)):
                data1.append([y_train[j], y_test[j]])
            if train_or_test == 'train':
                return np.array(data0)
            if train_or_test == 'val':
                return np.array(data1)
        else:
            data_file = TARGET_TEST
            with open(data_file, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)[1:]
                return np.array(data)


    def get_classes(self, data):
        classes = []

        for item in data:
            if item[1] not in classes:
                classes.append(item[1])

        return sorted(classes)

    def get_class_one_hot(self, class_str, train_or_test):
        if(train_or_test=='train'):
            label = self.classes_train.index(class_str)
            label = to_categorical(label, len(self.classes_train))
            return np.array(label)
        elif(train_or_test=='val'):
            label = self.classes_val.index(class_str)
            label = to_categorical(label, len(self.classes_val))
            return np.array(label)

    def image_generator(self, train_or_test):
        if train_or_test == 'train':
            data = self.train
        elif train_or_test == 'val':
            data = self.val

        print('\n\nCreating {} generator with {} samples.\n'.format(train_or_test, len(data)))

        while True:
            X, Y = [], []

            for _ in range(self.batch_size):
                sample = random.choice(data)

                image = self.preprocess_image(os.path.join(TARGET_TRAIN_DIR, sample[0]))

                X.append(image)
                Y.append(self.get_class_one_hot(sample[1],train_or_test))
            yield np.array(X), np.array(Y)

    def preprocess_image(self, image):
        x = load_img(image, target_size=self.image_shape[:2])
        x = img_to_array(x)
        x = preprocess_input(x)

        return x

    def load_sample(self, sample):
        # print(sample[0])
        image = self.preprocess_image(os.path.join(TARGET_TEST_DIR, sample[0]))
        image = np.expand_dims(image, axis=0)

        return image


