import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.datasets import cifar10, mnist

from parameters import *


def load_datasets(dataset):

    label_train = None
    label_test = None
    if dataset == 'cifar10':
        (image_train, label_train), (image_test, label_test) = cifar10.load_data()

        image_train = np.reshape(image_train, [-1, 32, 32, 3])
        image_test = np.reshape(image_test, [-1, 32, 32, 3])
        image_train = image_train.astype('float32') / 255
        image_test = image_test.astype('float32') / 255

        if args.grayscale:
            image_train = np.reshape(
                np.mean(image_train, axis=-1), (-1,) + input_shape)
            image_test = np.reshape(
                np.mean(image_test, axis=-1), (-1,) + input_shape)

            for idx, image in enumerate(image_train):
                image_train[idx] = equalize_histogram(image)

            for idx, image in enumerate(image_test):
                image_test[idx] = equalize_histogram(image)
    elif dataset == 'mnist':
        (image_train_t, label_train), (image_test_t, label_test) = mnist.load_data()

        image_train = np.zeros((60000, 32, 32, 1))
        image_test = np.zeros((10000, 32, 32, 1))

        image_train[:, 2:30, 2:30, 0] = image_train_t.astype('float32') / 255
        image_test[:, 2:30, 2:30, 0] = image_test_t.astype('float32') / 255

    elif dataset == 'ising':
        with open('/Users/andrew/Documents/rgml/ising_data/data_0_45.pkl', 'rb') as f:
            image_train = np.reshape(
                pickle.load(f)['data'], [-1, 81, 81, 2])
            image_train = image_train[:, 0:32, 0:32, :]
            image_train = image_train.astype(np.int32)
            image_train[image_train < 0] = 0
            image_train = np.reshape(image_train[:, :, :, 1], [-1, 32, 32, 1])

        image_test = image_train

    elif dataset == 'dimer':
        with open('/Users/andrew/Documents/rgml/dimer_data/dimer.pkl', 'rb') as f:
            image_train = np.reshape(pickle.load(f)['data'], [-1, 64, 64, 4])
            image_train = image_train[:, 0:32, 0:32, :]

        image_test = image_train

    elif dataset == 'test':
        with open('/Users/andrew/Documents/rgml/test_data/split.pkl', 'rb') as f:
            image_train = np.reshape(
                pickle.load(f)['data'], [-1, 32, 32, 2])
            image_train = np.reshape(image_train[:, :, :, 1], [-1, 32, 32, 1])

        image_test = image_train

    return image_train, label_train, image_test, label_test


def annealed_softmax(latent_dim, tau):
    def ret(logits):
        softmax_arg = logits / tau
        y = K.softmax(K.reshape(softmax_arg,
                                (-1, latent_dim)))
        return K.reshape(y, (-1, latent_dim))

    return ret


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=np.squeeze(y_test))
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    figure = np.zeros((32 * n, 32 * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(32, 32, 3)
            figure[i * 32: (i + 1) * 32,
                   j * 32: (j + 1) * 32, :] = digit

    plt.figure(figsize=(10, 10))
    start_range = 32 // 2
    end_range = n * 32 + start_range + 1
    pixel_range = np.arange(start_range, end_range, 32)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def equalize_histogram(image, number_bins=256):
    # get image histogram
    image_histogram, bins = np.histogram(
        image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = np.max(image) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


class AnnealingCallback(Callback):
    def __init__(self, var, schedule=None):
        self.var = var

        self.schedule = schedule

        if schedule == None:
            final = 1/1000
            self.schedule = [max(final, np.power(final, x/(epochs-5)))
                             for x in range(epochs)]

    def on_epoch_begin(self, epoch, logs={}):
        K.set_value(self.var, self.schedule[epoch])
