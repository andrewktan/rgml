import argparse

import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten, Input,
                          Lambda, Reshape)
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras.utils import plot_model

from vae_utils import *

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='VAE for CIFAR-10')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--optimizer', type=str, default='rmsprop')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--show_graphs', type=bool, default=False)

    args = parser.parse_args()

    # (hyper)parameters
    input_shape = (32, 32, 3)
    num_conv = 3
    intermediate_dim = 128
    latent_dim = 16
    num_filters = 32
    epochs = args.epochs
    batch_size = 128

    # import dataset
    (image_train, label_train), (image_test, label_test) = cifar10.load_data()

    image_train = np.reshape(image_train, [-1, 32, 32, 3])
    image_test = np.reshape(image_test, [-1, 32, 32, 3])
    image_train = image_train.astype('float32') / 255
    image_test = image_test.astype('float32') / 255

    # encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    x = Conv2D(filters=3,
               kernel_size=2,
               activation='relu',
               strides=1,
               padding='same')(x)

    x = Conv2D(filters=num_filters,
               kernel_size=2,
               activation='relu',
               strides=2,
               padding='same')(x)

    x = Conv2D(filters=num_filters,
               kernel_size=num_conv,
               activation='relu',
               strides=1,
               padding='same')(x)

    x = Conv2D(filters=num_filters,
               kernel_size=num_conv,
               activation='relu',
               strides=1,
               padding='same')(x)

    x_shape = K.int_shape(x)

    x = Flatten()(x)

    x = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(latent_dim,),
               name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)

    x = Dense(x_shape[1]*x_shape[2]*x_shape[3], activation='relu')(x)

    x = Reshape(x_shape[1:])(x)

    x = Conv2DTranspose(filters=num_filters,
                        kernel_size=num_conv,
                        activation='relu',
                        strides=1,
                        padding='same')(x)

    x = Conv2DTranspose(filters=num_filters,
                        kernel_size=num_conv,
                        activation='relu',
                        strides=1,
                        padding='same')(x)

    x = Conv2DTranspose(filters=num_filters,
                        kernel_size=(3, 3),
                        activation='relu',
                        strides=(2, 2),
                        padding='same')(x)

    outputs = Conv2D(filters=3,
                     kernel_size=(2, 2),
                     strides=1,
                     activation='sigmoid',
                     padding='same',
                     name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # cost function
    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                              K.flatten(outputs)) * 32**2
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=args.optimizer, loss=None)
    vae.summary()

    # plot architecture
    if args.show_graphs:
        plot_model(encoder, to_file='out/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='out/vae_decoder.png', show_shapes=True)
        plot_model(vae, to_file='out/vae.png', show_shapes=True)

    # train
    if args.train:
        vae.fit(image_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(image_test, None))

        vae.save_weights('store/vae_cifar_ld%03d_e%03d.h5' %
                         (latent_dim, epochs))
        encoder.save_weights('store/enc_cifar_ld%03d_e%03d.h5' %
                             (latent_dim, epochs))
        decoder.save_weights('store/dec_cifar_ld%03d_e%03d.h5' %
                             (latent_dim, epochs))
    else:
        vae.load_weights('store/vae_cifar_ld%03d_e%03d.h5' %
                         (latent_dim, epochs))

    for idx in range(5):
        img = vae.predict(
            np.reshape(
                image_test[idx, :, :, :], [1, 32, 32, 3]
            )
        )
        plt.imshow(np.concatenate((np.squeeze(img), image_test[idx, :, :, :])))
        plt.show()
