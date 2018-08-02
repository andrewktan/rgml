import argparse

import numpy as np
from keras.datasets import cifar10
from keras.layers import Lambda
from keras.losses import binary_crossentropy
from keras.utils import plot_model

from vae_components import *
from vae_utils import *

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='patch_encoder for CIFAR-10')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--load', dest='train', action='store_false')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')
    parser.add_argument('--show_graphs', dest='show_graphs',
                        action='store_true')
    parser.set_defaults(train=True, show_graphs=False, grayscale=False)

    args = parser.parse_args()

    # (hyper)parameters
    r = 15
    c = 15

    input_shape = (32, 32, 1) if args.grayscale else (32, 32, 3)
    hidden_dim = 32
    latent_dim = 128
    intermediate_dim = 256
    num_filters = 32
    num_conv = 4
    epochs = args.epochs
    beta = args.beta
    batch_size = 128

    # import dataset
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

    # patch encoder
    inputs = Input(shape=input_shape, name='encoder_input')

    x = Lambda(lambda x: x[:, r:r+4, c:c+4, :],
               output_shape=(4, 4, input_shape[2]))(inputs)

    encoder = Patch_Encoder(inputs,
                            hidden_dim=hidden_dim,
                            intermediate_dim=intermediate_dim,
                            latent_dim=latent_dim)

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

    # imaginer model
    [z_mean, z_log_var, z] = encoder(inputs)
    outputs = decoder(z)
    imaginer = Model(inputs, outputs, name='vae')

    # cost function
    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                              K.flatten(outputs)) * 32**2
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    imag_loss = K.mean(reconstruction_loss + beta * kl_loss)
    imaginer.add_loss(imag_loss)
    imaginer.compile(optimizer=args.optimizer, loss=None)
    imaginer.summary()

    # plot architecture
    if args.show_graphs:
        plot_model(encoder, to_file='out/vae_encoder.png', show_shapes=True)
        plot_model(decoder, to_file='out/vae_decoder.png', show_shapes=True)
        plot_model(imaginer, to_file='out/imaginer.png', show_shapes=True)

    # train
    decoder.load_weights("store/dec_cifar_ld%03d_e%03d.h5" %
                         (latent_dim, epochs))

    if args.train:
        imaginer.fit(image_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(image_test, None))

        imaginer.save_weights("store/imag_cifar_ld%03d_b%03d_e%03d.h5" %
                              (latent_dim, beta, epochs))
        encoder.save_weights("store/penc_cifar_ld%03d_b%03d_e%03d.h5" %
                             (latent_dim, beta, epochs))
    else:
        encoder.load_weights("store/penc_cifar_ld%03d_b%03d_e%03d.h5" %
                             (latent_dim, beta, epochs))

    for idx in range(10):
        img = imaginer.predict(
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
