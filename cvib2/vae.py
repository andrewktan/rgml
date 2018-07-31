import argparse

import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten, Input,
                          Lambda, Reshape)
from keras.losses import binary_crossentropy, mse
from keras.models import Model

from vae_utils import *

# from keras.utils import plot_model


# (hyper)parameters
input_shape = (32, 32, 3)
latent_dim = 16
epochs = 2
batch_size = 32


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
           kernel_size=(2, 2),
           activation='relu',
           strides=1,
           padding='same')(x)

x = Conv2D(filters=32,
           kernel_size=(2, 2),
           activation='relu',
           strides=2,
           padding='same')(x)

x = Conv2D(filters=32,
           kernel_size=(2, 2),
           activation='relu',
           strides=1,
           padding='same')(x)

x = Conv2D(filters=32,
           kernel_size=(2, 2),
           activation='relu',
           strides=1,
           padding='same')(x)

x = Flatten()(x)

x = Dense(4*4*32, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='out/vae_cnn_encoder.png', show_shapes=True)

# decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(128, activation='relu')(latent_inputs)

x = Dense(16*16*32, activation='relu')(x)

x = Reshape([16, 16, 32])(x)

x = Conv2DTranspose(filters=32,
                    kernel_size=(2, 2),
                    activation='relu',
                    strides=1,
                    padding='same')(x)

x = Conv2DTranspose(filters=32,
                    kernel_size=(2, 2),
                    activation='relu',
                    strides=1,
                    padding='same')(x)

x = Conv2DTranspose(filters=32,
                    kernel_size=(2, 2),
                    activation='relu',
                    strides=2,
                    padding='same')(x)

outputs = Conv2D(filters=3,
                 kernel_size=(2, 2),
                 strides=1,
                 activation='sigmoid',
                 padding='same',
                 name='decoder_output')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='out/vae_cnn_decoder.png', show_shapes=True)

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# plot_model(vae, to_file='out/vae.png', show_shapes=True)

if __name__ == '__main__':

    # cost function
    # reconstruction_loss = mse(K.flatten(inputs),
                              # K.flatten(outputs)) * 32**2
    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                              K.flatten(outputs)) * 32**2
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adagrad')
    vae.summary()

    # train
    vae.fit(image_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(image_test, None))

    # vae.load_weights('vae_cnn_cifar.h5')

    for idx in range(5):
        img = vae.predict(
            np.reshape(
                image_test[idx, :, :, :], [1, 32, 32, 3]
            )
        )
        plt.imshow(np.concatenate((np.squeeze(img), image_test[idx, :, :, :])))
        plt.show()
    # plot_results((encoder, decoder),
        # (image_test, label_test),
        # batch_size=batch_size,
        # model_name='vae')
