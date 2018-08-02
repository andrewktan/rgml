import argparse

import numpy as np
from keras.datasets import cifar10
from keras.layers import Lambda
from sklearn.cluster import KMeans

from vae_components import *

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser(description='patch_encoder for CIFAR-10')
    parser.add_argument('--num_clusters', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')

    parser.set_defaults(grayscale=False)

    args = parser.parse_args()

    # (hyper)parameters
    r = 15
    c = 15
    sz = 12

    input_shape = (32, 32, 1) if args.grayscale else (32, 32, 3)
    hidden_dim = 1024
    latent_dim = 128
    intermediate_dim = 256
    epochs = args.epochs
    beta = args.beta

    num_clusters = args.num_clusters

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

    x = Lambda(lambda x: x[:, r:r+sz, c:c+sz, :],
               output_shape=(sz, sz) + (input_shape[2],))(inputs)

    encoder = Patch_Encoder(inputs,
                            hidden_dim=hidden_dim,
                            intermediate_dim=intermediate_dim,
                            latent_dim=latent_dim)

    encoder.load_weights("store/penc_cifar_ld%03d_b%03d.h5" %
                         (latent_dim, beta))

    # encoder
    latents = encoder.predict(
        np.reshape(
            image_test, [-1, 32, 32, input_shape[2]]
        )
    )[0]

    # cluster
    cluster_id = KMeans(n_clusters=num_clusters).fit(latents).labels_

    receptive_fields = np.zeros((sz, sz*num_clusters, input_shape[2]))

    for cluster in range(num_clusters):
        receptive_fields[:, sz*cluster:sz*cluster+sz] = np.mean(
            image_test[cluster_id == cluster, r:r+sz, c:c+sz, :],
            axis=0)

        print(np.sum(cluster_id == cluster))

    plt.imshow(np.squeeze(receptive_fields),
               cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.show()
