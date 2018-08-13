import pickle

import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import Activation, Dense, Flatten
from keras.losses import binary_crossentropy, mse
from keras.models import Sequential
from keras.utils import plot_model, to_categorical

from parameters import *
from vae_utils import *

if __name__ == '__main__':
   # import dataset

    if False:
        (image_train, label_train), (image_test, label_test) = cifar10.load_data()
        image_train = np.reshape(image_train, (-1, 32, 32, 3))
        image_test = np.reshape(image_test, (-1, 32, 32, 3))
        image_train = image_train.astype('float32') / 255
        image_test = image_test.astype('float32') / 255
        if args.grayscale:
            image_train = np.reshape(
                np.mean(image_train, axis=-1), (-1,) + input_shape)
            image_test = np.reshape(
                np.mean(image_test, axis=-1), (-1,) + input_shape)
    else:
        with open("out/cifar_cg.pkl", 'rb') as f:
            dump = pickle.load(f)
        image_train = dump['train']['data']
        label_train = dump['train']['labels']
        image_test = dump['test']['data']
        label_test = dump['test']['labels']

    label_train = to_categorical(label_train, 10)
    label_test = to_categorical(label_test, 10)

    # linear classifier
    classifier = Sequential()
    classifier.add(Flatten())
    classifier.add(Dense(10))
    classifier.add(Activation('softmax'))

    classifier.compile(loss='categorical_crossentropy',
                       optimizer=args.optimizer,
                       metrics=['accuracy'])

    # plot architecture
    if args.show_graphs:
        plot_model(classifier, to_file='out/linear_classifier.png',
                   show_shapes=True)

    # train
    classifier.fit(image_train, label_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(image_test, label_test))

    classifier.summary()
