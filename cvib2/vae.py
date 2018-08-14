import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten, Input,
                          Lambda, Reshape)
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras.utils import plot_model

from parameters import *
from vae_components import *
from vae_utils import *

if __name__ == '__main__':
    # load datasets
    (image_train, label_train, image_test, label_test) = load_datasets(args.dataset)

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
                          num_channels=input_shape[2])

    decoder.summary()

    # VAE model
    [z_mean, z_log_var, z] = encoder(inputs)
    outputs = decoder(z)
    vae = Model(inputs, outputs, name='vae')

    # cost function
    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                              K.flatten(outputs)) * 32**2
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss)  # + kl_loss)
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

        vae.save_weights("store/vae_cifar_ld%03d_%d.h5" %
                         (latent_dim, input_shape[2]))
        encoder.save_weights("store/enc_cifar_ld%03d_%d.h5" %
                             (latent_dim, input_shape[2]))
        decoder.save_weights("store/dec_cifar_ld%03d_%d.h5" %
                             (latent_dim, input_shape[2]))
    else:
        vae.load_weights("store/vae_cifar_ld%03d_%d.h5" %
                         (latent_dim, input_shape[2]))

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
