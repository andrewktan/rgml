import argparse

import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten, Input,
                          Lambda, Reshape)
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras.utils import plot_model

from vae_components import *
from vae_utils import *

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='VAE for CIFAR-10')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--load', dest='train', action='store_false')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')
    parser.add_argument('--show_graphs', dest='show_graphs',
                        action='store_true')
    parser.set_defaults(train=True, show_graphs=False, grayscale=False)

    args = parser.parse_args()

    # (hyper)parameters
    input_shape = (32, 32, 1) if args.grayscale else (32, 32, 3)
    intermediate_dim = 256
    latent_dim = 128
    num_conv = 4
    num_filters = 32
    epochs = args.epochs
    batch_size = 128

    # import dataset
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

    # encoder
    inputs = Input(shape=input_shape, name='encoder_input')

    encoder = VAE_Encoder(inputs,
                          latent_dim=latent_dim,
                          intermediate_dim=intermediate_dim,
                          num_filters=num_filters,
                          num_conv=num_conv)

    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    decoder = VAE_Decoder(latent_inputs,
                          latent_dim=latent_dim,
                          intermediate_dim=intermediate_dim,
                          num_filters=num_filters,
                          num_conv=num_conv,
                          grayscale=args.grayscale)

    decoder.summary()

    # VAE model
    [z_mean, z_log_var, z] = encoder(inputs)
    outputs = decoder(z)
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

        vae.save_weights("store/vae_cifar_ld%03d_e%03d.h5" %
                         (latent_dim, epochs))
        encoder.save_weights("store/enc_cifar_ld%03d_e%03d.h5" %
                             (latent_dim, epochs))
        decoder.save_weights("store/dec_cifar_ld%03d_e%03d.h5" %
                             (latent_dim, epochs))
    else:
        vae.load_weights("store/vae_cifar_ld%03d_e%03d.h5" %
                         (latent_dim, epochs))

    for idx in range(10):
        img = vae.predict(
            np.reshape(
                image_test[idx], (1,) + input_shape
            )
        )
        plt.imshow(np.concatenate((np.squeeze(img),
                                   np.squeeze(image_test[idx]))
                                  ),
                   cmap=plt.cm.gray
                   )

        plt.show()
