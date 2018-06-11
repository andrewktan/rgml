import numpy as np
import matplotlib.pyplot as plt
from information_bottleneck import *
from ising_iterator import *

## parameters ##
################

dfile = '/Users/andrew/Documents/rgml/ising_data/data_0_50.txt'
sz = 25     # size of the samples (sq)
vsize = 3   # size of visible block (sq)
stride = 3
bsize = 3   # size of buffer (sq)
esize = 4   # environment size
tsize = 1000000   # table size

# load data #
#############

samples = IsingIterator(dfile)

# randomly choose environment #
# #############################

# env = np.empty((esize, 2), dtype=np.int32)
# for i in range(esize):
#     env[i,:] = np.random.randint(sz - vsize//2 - 2*bsize, size=2)

# symmetric environment choice
env = np.array([[-1,-1], [-1,1], [1,-1], [1,1]]) * bsize


# build joint distribution #
############################
table = np.empty((tsize, vsize*vsize + esize))
idx = 0

for sample in samples:

    sample = sample.reshape(sz,sz)

    for r in range(1, sz, stride):
        for c in range(1, sz, stride):
            if idx >= tsize:
                break

            rl = np.mod(r - vsize//2, sz)
            ru = np.mod(r + (vsize + 1)//2, sz)
            cl = np.mod(c - vsize//2, sz)
            cu = np.mod(c + (vsize + 1)//2, sz)

            if rl > ru or cl > cu:  # hacky fix to wraparound
                continue

            table[idx,0:vsize*vsize] = np.reshape(sample[rl:ru,cl:cu],-1)
            for k in range(esize):
                esamp = np.mod(np.array((r,c)) + env[k,:],sz)
                table[idx,-(k+1)] = sample[esamp[0], esamp[1]]

            idx += 1


def row2bin(x):
    a = 0
    b = 0
    for i,j in enumerate(x):
        j = 1 if j == 1 else 0

        if i < vsize*vsize:
            a += j<<i
        else:
            b += j<<(i-vsize*vsize)
    return np.array([a, b])

table2 = np.apply_along_axis(row2bin, 1, table)

#plt.matshow(table[0:20,:], cmap=plt.cm.gray)
#plt.show()

# lolhistogram
thist = np.zeros((2**(vsize*vsize), 2**(esize)))

for r,c in table2:
    thist[r,c] += 1

thist /= tsize

#plt.matshow(pxy, cmap=plt.cm.gray)
#plt.show()

# information bottleneck #
##########################

dib = DIB(thist, beta=5, hiddens=100)
dib.compress()
dib.report_clusters()
dib.visualize_clusters()
