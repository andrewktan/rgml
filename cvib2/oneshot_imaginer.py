import pickle

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Lambda
from keras.losses import binary_crossentropy
from keras.utils import plot_model

from parameters import *
from vae_components import *
from vae_utils import *

if __name__ == '__main__':
    # load datasets
    (image_train, label_train, image_test, label_test) = load_datasets(args.dataset)

    # patch encoder
    inputs = Input(shape=input_shape, name='encoder_input')

    encoder = Patch_Encoder(inputs, r, c, sz,
                            hidden_dim=hidden_dim,
                            intermediate_dim=intermediate_dim,
                            latent_dim=latent_dim)

    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    if args.dataset == 'cifar10':
        decoder = VAE_Decoder(latent_inputs,
                              latent_dim=latent_dim,
                              intermediate_dim=intermediate_dim,
                              num_filters=num_filters,
                              num_conv=num_conv,
                              num_channels=input_shape[2])
    else:
        decoder = VAE_Decoder_NC(latent_inputs,
                                 latent_dim=latent_dim,
                                 intermediate_dim=intermediate_dim,
                                 num_channels=input_shape[2])

    decoder.summary()

    # imaginer model
    [z_mean, z_log_var, z] = encoder(inputs)
    outputs = decoder(z)
    imaginer = Model(inputs, outputs, name='vae')

    # cost function
    def mask(x):
        m = np.ones(input_shape[0:2], dtype=np.bool)
        m[r-2:r+sz+2, c-2:c+sz+2] = False
        m[r:r+sz, c:c+sz] = False

        x = tf.transpose(x, perm=[1, 2, 0])
        x = tf.boolean_mask(x, m)
        x = tf.transpose(x)

        return x

# cost function
    # reconstruction_loss = binary_crossentropy(K.flatten(inputs),
    #                                           K.flatten(outputs)) * \
    #     input_shape[0]*input_shape[1]
    reconstruction_loss = inputs * \
        K.log(outputs + 1e-12) + (1-inputs) * K.log(1-outputs + 1e-12)
    reconstruction_loss = K.sum(reconstruction_loss, axis=3)
    reconstruction_loss = K.mean(mask(reconstruction_loss))
    reconstruction_loss *= -1

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(K.sum(kl_loss, axis=-1))
    kl_loss *= - 0.5

    imag_loss = kl_loss + beta * reconstruction_loss
    imaginer.add_loss(imag_loss)
    imaginer.compile(optimizer=args.optimizer, loss=None)
    imaginer.summary()

    # plot architecture
    if args.show_graphs:
        plot_model(encoder, to_file='out/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='out/vae_decoder.png', show_shapes=True)
        plot_model(imaginer, to_file='out/imaginer.png', show_shapes=True)

    # train
    if args.train:
        imaginer.fit(image_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(image_test, None))

        imaginer.save_weights("store/imag_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                              (args.dataset, latent_dim, beta, r, c, input_shape[2]))
        encoder.save_weights("store/penc_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                             (args.dataset, latent_dim, beta, r, c, input_shape[2]))
    else:
        imaginer.load_weights("store/imag_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                              (args.dataset, latent_dim, beta, r, c, input_shape[2]))

    for idx in range(10):
        img = imaginer.predict(
            np.reshape(
                image_test[idx], (1,) + input_shape
            )
        )

        if args.dataset == 'cifar10':
            plt.imshow(np.concatenate((np.squeeze(img),
                                       np.squeeze(image_test[idx]))
                                      ),
                       cmap=plt.cm.gray
                       )
        elif args.dataset == 'dimer':
            actual_image = np.squeeze(np.argmax(image_test[idx], axis=-1))

            predicted_image = np.squeeze(img[:, :, :, 1] +
                                         img[:, :, :, 2]*2 +
                                         img[:, :, :, 3]*3)

            loss_image = image_test[idx] * \
                np.log(img[0]) + (1-image_test[idx]+1e-12) * \
                np.log(1-img[0]+1e-12)

            loss_image *= -1
            loss_image = np.sum(loss_image, axis=2)

            plt.imshow(np.concatenate((predicted_image, actual_image)),
                       cmap=plt.cm.gray
                       )
        elif args.dataset == 'ising' or args.dataset == 'test':
            plt.imshow(np.concatenate((np.squeeze(img[:, :, :, 1]),
                                       np.squeeze(np.argmax(image_test[idx], axis=-1)))
                                      ),
                       cmap=plt.cm.gray
                       )

        plt.show()
