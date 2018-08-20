import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
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

    encoder = Patch_Encoder_D(inputs, r, c, sz,
                              hidden_dim=hidden_dim,
                              intermediate_dim=intermediate_dim,
                              latent_dim=latent_dim)

    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='latent_inputs')
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
    z = encoder(inputs)

    # gumbel reparametrization and annealing
    class AnnealingCallback(Callback):
        def __init__(self, var, schedule=None):
            self.var = var

            self.schedule = schedule

            if schedule == None:
                decay = np.power(1/10, 1/(epochs-1))
                self.schedule = [np.power(decay, x) for x in range(epochs)]

        def on_epoch_begin(self, epoch, logs={}):
            K.set_value(self.var, self.schedule[epoch])

    tau = K.variable(1.)

    z_samp = Lambda(gumbel_softmax(latent_dim, tau=tau))(z)

    outputs = decoder(z_samp)
    imaginer = Model(inputs, outputs, name='imaginer_d')

    # cost function
    def mask(x):
        m = np.ones(input_shape[0:2], dtype=np.bool)
        m[r-5:r+sz+5, c-5:c+sz+5] = True
        m[r:r+sz, c:c+sz] = False

        x = tf.transpose(x, perm=[1, 2, 0])
        x = tf.boolean_mask(x, m)
        x = tf.transpose(x)

        return x

    # inputs_masked = Lambda(mask)(inputs)
    # outputs_masked = Lambda(mask)(outputs)

    # reconstruction_loss = binary_crossentropy(K.flatten(inputs_masked),
        # K.flatten(outputs_masked)) * 32**2

    reconstruction_loss = inputs * \
        K.log(outputs + K.epsilon()) + (1-inputs) * \
        K.log(1-outputs + K.epsilon())
    reconstruction_loss = K.sum(reconstruction_loss, axis=3)
    reconstruction_loss = K.mean(mask(reconstruction_loss))
    reconstruction_loss *= -1

    pz = K.mean(z, axis=0)

    kl_loss = - K.sum(pz * K.log(pz + K.epsilon()), axis=-1)

    imag_loss = kl_loss + beta * reconstruction_loss
    imag_loss /= np.log(2)

    imaginer.add_loss(imag_loss)
    imaginer.add_loss(reconstruction_loss)
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
                     validation_data=(image_test, None),
                     callbacks=[AnnealingCallback(tau)])

        imaginer.save_weights("store/imag_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                              (args.dataset, latent_dim, beta, r, c, input_shape[2]))
        encoder.save_weights("store/penc_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                             (args.dataset, latent_dim, beta, r, c, input_shape[2]))
    else:
        imaginer.load_weights("store/imag_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                              (args.dataset, latent_dim, beta, r, c, input_shape[2]))

        K.set_value(tau, 1/100)

    for idx in range(10):
        img = imaginer.predict(
            np.reshape(
                image_test[idx], (1,) + input_shape
            )
        )

        latents = encoder.predict(
            np.reshape(
                image_test[idx], (1,)+input_shape
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

            plt.imshow(np.concatenate((predicted_image, actual_image)),
                       cmap=plt.cm.gray
                       )
        elif args.dataset == 'ising' or args.dataset == 'test':
            plt.imshow(np.concatenate((np.squeeze(img),
                                       np.squeeze(image_test[idx]))
                                      ),
                       cmap=plt.cm.gray
                       )

            print(latents[0, 21], latents[0, 28])
        plt.show()
