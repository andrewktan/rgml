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
    # import dataset
    (image_train, label_train), (image_test, label_test) = cifar10.load_data()

    image_train = np.reshape(image_train, [-1, 32, 32, 3])
    image_test = np.reshape(image_test, [-1, 32, 32, 3])
    image_train = image_train.astype('float32') / 255
    image_test = image_test.astype('float32') / 255

    # with open('/Users/andrew/Documents/rgml/test_data/split/data.pkl', 'rb') as f:
    # image_train = np.reshape(pickle.load(f)['data'], [-1, 32, 32, 1])

    # image_test = image_train

    if args.grayscale:
        image_train = np.reshape(
            np.mean(image_train, axis=-1), (-1,) + input_shape)
        image_test = np.reshape(
            np.mean(image_test, axis=-1), (-1,) + input_shape)

    # patch encoder
    inputs = Input(shape=input_shape, name='encoder_input')

    encoder = Patch_Encoder(inputs, r, c, sz,
                            hidden_dim=hidden_dim,
                            intermediate_dim=intermediate_dim,
                            latent_dim=latent_dim,
                            deterministic=True)

    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='latent_inputs')
    decoder = VAE_Decoder(latent_inputs,
                          latent_dim=latent_dim,
                          intermediate_dim=intermediate_dim,
                          num_filters=num_filters,
                          num_conv=num_conv,
                          grayscale=args.grayscale)

    decoder.summary()

    # imaginer model
    z = encoder(inputs)
    outputs = decoder(z)
    imaginer = Model(inputs, outputs, name='imaginer_d')

    # cost function
    def mask(x):
        m = np.zeros(input_shape, dtype=np.bool)
        m[r-4:r+sz+4, c-4:c+sz+4, :] = True
        m[r:r+sz, c:c+sz, :] = False

        x = tf.transpose(x, perm=[1, 2, 3, 0])
        x = tf.boolean_mask(x, m)
        x = tf.transpose(x)

        return x

    inputs_masked = Lambda(mask)(inputs)
    outputs_masked = Lambda(mask)(outputs)

    reconstruction_loss = binary_crossentropy(K.flatten(inputs_masked),
                                              K.flatten(outputs_masked)) * 32**2
    kl_loss = z * K.log(z + 1e-12)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= 0.5

    imag_loss = K.mean(beta * reconstruction_loss - kl_loss)
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

        imaginer.save_weights("store/imag_cifar_ld%03d_b%03d_%d.h5" %
                              (latent_dim, beta, input_shape[2]))
        encoder.save_weights("store/penc_cifar_ld%03d_b%03d_%d.h5" %
                             (latent_dim, beta, input_shape[2]))
    else:
        imaginer.load_weights("store/imag_cifar_ld%03d_b%03d_%d.h5" %
                              (latent_dim, beta, input_shape[2]))

    for idx in range(10):
        img = imaginer.predict(
            np.reshape(
                image_test[idx], (1,) + input_shape
            )
        )

        plt.imshow(np.concatenate((np.squeeze(img[:, r-4:r+sz+4, c-4:c+sz+4, :]),
                                   np.squeeze(image_test[idx, r-4:r+sz+4, c-4:c+sz+4, :]))
                                  ),
                   cmap=plt.cm.gray
                   )

        plt.show()
