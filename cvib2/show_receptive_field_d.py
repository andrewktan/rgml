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

    encoder = Patch_Encoder_D(inputs, r, c, sz,
                              hidden_dim=hidden_dim,
                              intermediate_dim=intermediate_dim,
                              latent_dim=latent_dim)

    encoder.load_weights("store/penc_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                         (args.dataset, latent_dim, beta, r, c, input_shape[2]))

    # encoder
    latents = encoder.predict(
        np.reshape(
            image_test, (-1,) + input_shape
        )
    )

    # cluster
    cluster_id = np.argmax(latents, axis=-1)

    num_clusters = 0

    for ld in range(latent_dim):
        if np.sum(cluster_id == ld) > 0:
            num_clusters += 1

    receptive_fields = np.zeros((sz, sz*num_clusters))

    cluster = 0
    for ld in range(latent_dim):
        print(np.sum(cluster_id == ld))
        if np.sum(cluster_id == ld) == 0:
            continue

        if args.dataset == 'cifar10':
            receptive_fields[:, sz*cluster:sz*cluster+sz] = np.squeeze(
                np.mean(
                    image_test[cluster_id == ld, r:r+sz, c:c+sz, :],
                    axis=0))
        elif args.dataset == 'dimer':
            receptive_fields[:, sz*cluster:sz*cluster+sz] = np.mean(
                image_test[cluster_id == ld, r:r+sz, c:c+sz, 1] +
                2*image_test[cluster_id == ld, r:r+sz, c:c+sz, 2] +
                3*image_test[cluster_id == ld, r:r+sz, c:c+sz, 3],
                axis=0)
        elif args.dataset == 'ising' or args.dataset == 'test':
            receptive_fields[:, sz*cluster:sz*cluster+sz] = np.mean(
                image_test[cluster_id == ld, r:r+sz, c:c+sz, 1],
                axis=0)

        cluster += 1

    plt.imshow(np.squeeze(receptive_fields),
               cmap=plt.cm.gray)

    plt.show()
