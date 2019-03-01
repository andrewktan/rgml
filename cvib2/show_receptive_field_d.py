import pickle

import numpy as np
from keras.datasets import cifar10
from keras.layers import Lambda

from parameters import *
from vae_components import *

if __name__ == '__main__':
    # load datasets
    (image_train, label_train, image_test, label_test) = load_datasets(args.dataset)

    # patch encoder
    inputs = Input(shape=input_shape, name='encoder_input')

    encoder, tau = Patch_Encoder_D(inputs, r, c, sz,
                                   hidden_dim=hidden_dim,
                                   intermediate_dim=intermediate_dim,
                                   latent_dim=latent_dim)

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='latent_inputs')
    if args.dataset == 'cifar10' and False:
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

    z = encoder(inputs)
    outputs = decoder(z)
    imaginer = Model(inputs, outputs, name='imaginer_d')

    imaginer.load_weights("store/imag_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                          (args.dataset, latent_dim, beta, r, c, input_shape[2]))

    K.set_value(tau, 1e-12)

    latents = encoder.predict(
        np.reshape(
            image_test, (-1,) + input_shape
        )
    )

    dec_images = imaginer.predict(
        np.reshape(
            image_test, (-1,) + input_shape
        )
    )

    # cluster
    cluster_id = np.argmax(latents, axis=-1)

    cluster_pops = [np.sum(cluster_id == ld) for ld in range(latent_dim)]
    cluster_pops = np.array(cluster_pops)
    cluster_order = np.argsort(-cluster_pops)
    cluster_pops = cluster_pops[cluster_pops > 100]

    num_clusters = cluster_pops.size

    receptive_fields = np.zeros((sz, sz*num_clusters))

    cluster = 0

    disp = np.zeros((32, 32*num_clusters, 3))

    for idx in range(num_clusters):
        ld = cluster_order[idx]

        print(np.sum(cluster_id == ld))

        if args.dataset == 'cifar10' or args.dataset == 'mnist' or args.dataset == 'ising':
            cluster_rf = np.squeeze(
                np.mean(
                    image_test[cluster_id == ld, r:r+sz, c:c+sz, :],
                    axis=0)
            )

            receptive_fields[:, sz*cluster:sz*cluster+sz] = cluster_rf

            output = np.squeeze(
                np.mean(
                    dec_images[cluster_id == ld],
                    axis=0)
            )

            output[r:r+sz, c:c+sz] = 0
            disp[0:32, idx*32:(idx+1)*32, 0] = output
            disp[0:32, idx*32:(idx+1)*32, 1] = output
            disp[0:32, idx*32:(idx+1)*32, 2] = output
            disp[r:r+sz, idx*32+c:idx*32+c+sz,
                 1] = cluster_rf

        elif args.dataset == 'dimer':
            receptive_fields[:, sz*cluster:sz*cluster+sz] = np.mean(
                image_test[cluster_id == ld, r:r+sz, c:c+sz, 1] +
                2*image_test[cluster_id == ld, r:r+sz, c:c+sz, 2] +
                3*image_test[cluster_id == ld, r:r+sz, c:c+sz, 3],
                axis=0)

        cluster += 1

    plt.imshow(np.squeeze(disp),
               cmap=plt.cm.gray)

    plt.show()
