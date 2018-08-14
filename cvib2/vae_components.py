import numpy as np
from keras import Model
from keras import backend as K
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Flatten, Input, Lambda, Reshape, Softmax)

from vae_utils import *


def VAE_Encoder(inputs,
                latent_dim=16,
                intermediate_dim=128,
                num_filters=32,
                num_conv=4,
                name='encoder',
                grayscale=False):
    # build encoder
    layers = [Conv2D(filters=1 if grayscale else 3,
                     kernel_size=2,
                     activation='relu',
                     strides=1,
                     padding='same'),
              Conv2D(filters=2,
                     kernel_size=2,
                     activation='relu',
                     strides=2,
                     padding='same'),
              Conv2D(filters=num_filters,
                     kernel_size=num_conv,
                     activation='relu',
                     strides=1,
                     padding='same'),
              Conv2D(filters=num_filters,
                     kernel_size=num_conv,
                     activation='relu',
                     strides=1,
                     padding='same'),
              Flatten(),
              Dense(intermediate_dim, activation='relu')]

    # connect everything
    x = inputs

    for layer in layers:
        x = layer(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,),
               name='z')([z_mean, z_log_var])

    return Model(inputs, [z_mean, z_log_var, z], name=name)


def Patch_Encoder(inputs, r, c, sz,
                  hidden_dim=32,
                  intermediate_dim=128,
                  latent_dim=16,
                  name='patch_encoder'):

    # build encoder
    layers = [Lambda(lambda x: x[:, r:r+sz, c:c+sz, :]),
              Flatten(),
              Dense(hidden_dim,
                    activation='relu'),
              Dense(hidden_dim,
                    activation='relu'),
              Dense(hidden_dim,
                    activation='relu'),
              Dense(intermediate_dim,
                    activation='relu')]

    # connect everything
    x = inputs

    for layer in layers:
        x = layer(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,),
               name='z')([z_mean, z_log_var])

    return Model(inputs, [z_mean, z_log_var, z], name=name)


def Patch_Encoder_D(inputs, r, c, sz,
                    hidden_dim=32,
                    intermediate_dim=128,
                    latent_dim=16,
                    name='patch_encoder_d'):

    # build encoder
    layers = [Lambda(lambda x: x[:, r:r+sz, c:c+sz, :]),
              Flatten(),
              Dense(hidden_dim,
                    activation='elu'),
              BatchNormalization(),
              Dense(hidden_dim,
                    activation='elu'),
              BatchNormalization(),
              Dense(latent_dim, activation='elu'),
              Softmax()]

    # connect everything
    x = inputs

    for layer in layers:
        x = layer(x)

    z_det = x

    return Model(inputs, z_det)


def VAE_Decoder(inputs,
                latent_dim=16,
                intermediate_dim=128,
                num_filters=32,
                num_conv=4,
                num_channels=1,
                name='decoder'):

    # build decoder
    x_shape = (16, 16, 32)

    layers = [Dense(intermediate_dim, activation='relu'),
              Dense(np.prod(x_shape), activation='relu'),
              Reshape(x_shape),
              Conv2DTranspose(filters=num_filters,
                              kernel_size=num_conv,
                              activation='relu',
                              strides=1,
                              padding='same'),
              Conv2DTranspose(filters=num_filters,
                              kernel_size=num_conv,
                              activation='relu',
                              strides=1,
                              padding='same'),
              Conv2DTranspose(filters=num_filters,
                              kernel_size=num_conv,
                              activation='relu',
                              strides=2,
                              padding='same'),
              Conv2DTranspose(filters=num_channels,
                              kernel_size=2,
                              activation='sigmoid',
                              strides=1,
                              padding='same')]

    # connect everything
    x = inputs

    for layer in layers:
        x = layer(x)

    outputs = x

    return Model(inputs, outputs, name=name)


def VAE_Decoder_NC(inputs,
                   latent_dim=16,
                   intermediate_dim=128,
                   num_channels=1,
                   name='decoder'):

    # build decoder
    layers = [
        Dense(intermediate_dim, activation='elu'),
        BatchNormalization(),
        Dense(intermediate_dim, activation='elu'),
        BatchNormalization(),
        Dense(32*32*num_channels, activation='elu'),
        Reshape([32, 32, num_channels]),
        Softmax(axis=3)
    ]

    # connect everything
    x = inputs

    for layer in layers:
        x = layer(x)

    outputs = x

    return Model(inputs, outputs, name=name)
