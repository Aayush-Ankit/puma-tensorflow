
import gzip
import numpy as np
from time import time
from puma_layers import puma_dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

import seaborn as sns
from requests import get
sns.set()


def download_file(url, file_name):

    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)


def download_minist_files():
    download_file('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz')
    download_file('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    download_file('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    download_file('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


def read_mnist(images_path: str, labels_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path,'rb') as imagesFile:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
            .reshape(length, 784) \
            .reshape(length, 28, 28, 1)

    return features, labels


def get_features():
    train = {}
    train['features'], train['labels'] = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    test = {}
    test['features'], test['labels'] = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

    validation = {}
    train['features'], validation['features'], train['labels'], validation['labels'] = train_test_split(train['features'], train['labels'], test_size=0.2, random_state=0)
    train['features']      = np.pad(train['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
    validation['features'] = np.pad(validation['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
    test['features']       = np.pad(test['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
    return train['features'], train['labels'], validation['features'], validation['labels'], test['features'], test['labels']


def get_model(weights=None) :
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(units=84, activation='relu'))
    model.add(layers.Dense(units=10, activation = 'softmax'))
    if weights is not None:
        model.load_weights(weights)

    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model

def get_puma_model(weights=None):
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())

    #test 1:
    model.add(puma_dense(units=120, input_slices=[[[0, 0], [-1, 576]], [[0, 0], [-1, 576]], [[0, 0], [-1, 576]]],
                         kernel_slices=[[[0, 0], [-1, 40]], [[0, 40], [-1, 40]], [[0, 80], [-1, 40]]]))

    model.add(layers.Activation("relu"))
    # test 2:
    # model.add(layers.Dense(units=84, activation='relu'))
    model.add(puma_dense(units=84, input_slices=[[[0, 0], [-1, 60]], [[0, 60], [-1, 60]], [[0, 0], [-1, 60]], [[0, 60], [-1, 60]]],
                         kernel_slices=[[[0, 0], [60, 42]], [[60, 0], [60, 42]], [[0, 42], [60, 42]], [[60, 42], [60, 42]]]))
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(units=10, activation = 'softmax'))
    if weights is not None:
        model.load_weights(weights)

    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model

def train(m, train_f, train_l, validation_f, validation_l, weights=None):


    EPOCHS = 10
    BATCH_SIZE = 128

    X_train, y_train = train_f, to_categorical(train_l)

    X_validation, y_validation = validation_f, to_categorical(validation_l)
    train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
    validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)

    print('# of training images:', train_f.shape[0])
    print('# of validation images:', validation_f.shape[0])

    steps_per_epoch = X_train.shape[0]//BATCH_SIZE
    validation_steps = X_validation.shape[0]//BATCH_SIZE

    m.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                                        validation_data=validation_generator, validation_steps=validation_steps,
                                        shuffle=True)
    if weights is not None:
        m.save_weights(weights, overwrite=True)


def evaluate(m, test_f, test_l):
    score = m.evaluate(test_f, to_categorical(test_l))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
