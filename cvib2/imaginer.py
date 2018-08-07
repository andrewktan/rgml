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

    # decoder.trainable = False
    decoder.summary()

    # imaginer model
    [z_mean, z_log_var, z] = encoder(inputs)
    outputs = decoder(z)
    imaginer = Model(inputs, outputs, name='vae')

    # cost function
    def mask(x):
        m = np.ones(input_shape, dtype=np.bool)
        m[r:r+sz, c:c+sz] = False

        return tf.boolean_mask(x, m, axis=1)

    inputs_masked = Lambda(mask)(inputs)
    outputs_masked = Lambda(mask)(outputs)

    reconstruction_loss = binary_crossentropy(inputs_masked,
                                              outputs_masked) * (
        input_shape[0]*input_shape[1] - sz**2)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(kl_loss, axis=-1)
    kl_loss *= -0.5

    imag_loss = K.mean(reconstruction_loss + beta * kl_loss)
    imaginer.add_loss(imag_loss)
    imaginer.compile(optimizer=args.optimizer, loss=None)
    imaginer.summary()

    # plot architecture
    if args.show_graphs:
        plot_model(encoder, to_file='out/vae_encoder.png',
                   show_shapes=True)
        plot_model(decoder, to_file='out/vae_decoder.png',
                   show_shapes=True)
        plot_model(imaginer, to_file='out/imaginer.png', show_shapes=True)

    # train
    decoder.load_weights("store/dec_cifar_ld%03d_%d.h5" %
                         (latent_dim, input_shape[2]))

    if args.train:
        imaginer.fit(image_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(image_test, None))

        imaginer.save_weights("store/imag_cifar_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                              (latent_dim, beta, r, c, input_shape[2]))
        encoder.save_weights("store/penc_cifar_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                             (latent_dim, beta, r, c, input_shape[2]))
    else:
        encoder.load_weights("store/penc_cifar_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                             (latent_dim, beta, r, c, input_shape[2]))

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
