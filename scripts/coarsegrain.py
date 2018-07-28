import pickle

import matplotlib.pyplot as plt
import numpy as np

from cifar_iterator import *
from ising_iterator import *

# parameters
eloc = 3
beta = 15
vsz = 3
sz = 32
nsamp = 10000

dpath = '/Users/andrew/Documents/rgml/cifar-10_data/'

orig_name = 'test_batch'
dest_name = "%s_cg_%02d_%02d" % (orig_name, eloc, beta)

cg_file = "out/cg_%02d_%02d.pkl" % (eloc, beta)

# load data
with open(dpath + orig_name, 'rb') as f:
    dump = pickle.load(f, encoding='bytes')
    samples = dump[b'data']
    labels = dump[b'labels']

    # binarize data
    samples = samples.reshape(-1, 3, sz, sz)
    samples = np.mean(samples, axis=1) / 255
    samples = samples.reshape(-1, sz ** 2)
    samples = samples > 0.5
    samples = samples.astype(np.int32)

with open(cg_file, 'rb') as f:
    cg = pickle.load(f)
    print("Loaded coarse-graining file: %s" % cg_file)

# helpful functions


def to_code(spins):
    ret = 0
    for i, j in enumerate(spins):
        j = 1 if j == 1 else 0
        ret += j << i

    return ret


# perform coarse graining

cg_samples = np.empty((nsamp, 100))

for idx, sample in enumerate(samples):
    if idx % 1000 == 0:
        print(idx)
    sample = sample.reshape(sz, sz)
    sample_cg = np.empty((10, 10))
    for r in range(1, sz-1, vsz):
        for c in range(1, sz-1, vsz):
            vcode = to_code(np.reshape(sample[r:r+vsz, c:c+vsz], -1))
            sample_cg[r//vsz, c//vsz] = cg['f'][vcode]

    sample_cg = np.reshape(sample_cg, (1, -1))
    cg_samples[idx, :] = sample_cg

with open(dpath + dest_name, 'wb') as f:
    dump = {}
    dump[b'data'] = cg_samples
    dump[b'labels'] = labels

    pickle.dump(dump, f)
