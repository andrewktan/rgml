import pickle

import numpy as np
from keras.layers import Lambda
from sklearn.cluster import KMeans

from parameters import *
from vae_components import *
from vae_utils import *

if __name__ == '__main__':
    # load datasets
    (image_train, label_train, image_test, label_test) = load_datasets(args.dataset)

    # patch encode
    inputs = Input(shape=input_shape, name='encoder_input')

    encoder = Patch_Encoder(inputs, r, c, sz,
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
    )[0]

    # cluster
    cluster_id = KMeans(n_clusters=num_clusters).fit(latents).labels_

    receptive_fields = np.zeros((sz, sz*num_clusters))

    # rank clusters and relabel
    cluster_pop = np.zeros(num_clusters)
    for cluster in range(num_clusters):
        cluster_pop[cluster] = np.sum(cluster_id == cluster)

    cluster_map = np.argsort(-cluster_pop)

    print(cluster_map)

    cluster_id = np.apply_along_axis(lambda x: cluster_map[x],
                                     axis=-1,
                                     arr=cluster_id)

    for cluster in range(num_clusters):
        if args.dataset == 'cifar10':
            receptive_fields[:, sz*cluster:sz*cluster+sz] = np.mean(
                image_test[cluster_id == cluster, r:r+sz, c:c+sz, :],
                axis=0)
        elif args.dataset == 'dimer':
            receptive_fields[:, sz*cluster:sz*cluster+sz] = np.mean(
                image_test[cluster_id == cluster, r:r+sz, c:c+sz, 1] +
                2*image_test[cluster_id == cluster, r:r+sz, c:c+sz, 2] +
                3*image_test[cluster_id == cluster, r:r+sz, c:c+sz, 3],
                axis=0)
        elif args.dataset == 'ising' or args.dataset == 'test':
            receptive_fields[:, sz*cluster:sz*cluster+sz] = np.mean(
                image_test[cluster_id == cluster, r:r+sz, c:c+sz, 1],
                axis=0)

    plt.imshow(np.squeeze(receptive_fields),
               cmap=plt.cm.gray)

    plt.show()
