import pickle

import numpy as np
from keras.datasets import cifar10
from keras.layers import Lambda

from parameters import *
from vae_components import *

if __name__ == '__main__':
    # load datasets
    (image_train, label_train, image_test, label_test) = load_datasets(args.dataset)

    # load patch encoder grid
    enc_rows = input_shape[0] // sz
    enc_cols = input_shape[1] // sz

    num_clusters = latent_dim
    eye = np.eye(num_clusters)
    image_train_cg = np.zeros(
        (image_train.shape[0], enc_rows, enc_cols, num_clusters))
    image_test_cg = np.zeros(
        (image_test.shape[0], enc_rows, enc_cols, num_clusters))

    for r in range(enc_rows):
        for c in range(enc_cols):
            print(r, c)

            inputs = Input(shape=input_shape, name='encoder_input')

            # build and load patch encoder
            encoder, _ = Patch_Encoder_D(inputs, r*sz, c*sz, sz,
                                         hidden_dim=hidden_dim,
                                         intermediate_dim=intermediate_dim,
                                         latent_dim=latent_dim)

            encoder.load_weights("store/penc_%s_ld%03d_b%03d_r%02d_c%02d_%d.h5" %
                                 (args.dataset, latent_dim, beta, r*sz, c*sz, input_shape[2]))

            # coarsegrain
            latents_train = encoder.predict(
                np.reshape(
                    image_train, (-1,) + input_shape)
            )

            latents_test = encoder.predict(
                np.reshape(
                    image_test, (-1,) + input_shape)
            )

            image_train_cg[:, r, c, :] = eye[np.argmax(latents_train, axis=-1)]
            image_test_cg[:, r, c, :] = eye[np.argmax(latents_test, axis=-1)]

    # save data
    dump = {}
    dump['train'] = {}
    dump['test'] = {}

    dump['train']['data'] = image_train_cg
    dump['train']['labels'] = label_train
    dump['test']['data'] = image_test_cg
    dump['test']['labels'] = label_test

    with open("out/cifar_cg.pkl", 'wb') as f:
        pickle.dump(dump, f)
