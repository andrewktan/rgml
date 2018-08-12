import pickle

import numpy as np
from keras.datasets import cifar10
from keras.layers import Lambda

from parameters import *
from vae_components import *

if __name__ == '__main__':
    # import dataset
    (image_train, label_train), (image_test, label_test) = cifar10.load_data()

    image_train = np.reshape(image_train, [-1, 32, 32, 3])
    image_test = np.reshape(image_test, [-1, 32, 32, 3])
    image_train = image_train.astype('float32') / 255
    image_test = image_test.astype('float32') / 255

    # with open('/Users/andrew/Documents/rgml/test_data/split/data.pkl', 'rb') as f:
    #     image_train = np.reshape(pickle.load(f)['data'], [-1, 32, 32, 1])

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

    encoder.load_weights("store/penc_cifar_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                         (latent_dim, beta, r, c, input_shape[2]))

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

    receptive_fields = np.zeros((sz, sz*num_clusters, input_shape[2]))

    cluster = 0
    for ld in range(latent_dim):
        if np.sum(cluster_id == ld) == 0:
            continue

        receptive_fields[:, sz*cluster:sz*cluster+sz] = np.mean(
            image_test[cluster_id == ld, r:r+sz, c:c+sz, :],
            axis=0)

        cluster += 1

    plt.imshow(np.squeeze(receptive_fields),
               cmap=plt.cm.gray)
    plt.show()
