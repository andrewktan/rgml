import matplotlib.pyplot as plt
import numpy as np
from information_bottleneck import *
from ising_iterator import *

# parameters #
##############

perform_beta_sweep = False
perform_demo = True

dfile = '/Users/andrew/Documents/rgml/ising_data/data_0_50.txt'
savefile = 'ising_ib_joint.npy'
sz = 25     # size of the samples (sq)
vsize = 3   # size of visible block (sq)
stride = 3
bsize = 1   # size of buffer (sq)
esize = 4   # environment size
tsize = 1000000   # table size

# load data #
#############


def calculate_joint():
    samples = IsingIterator(dfile)

    # choose environment #
    # env = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * bsize
    env = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]]) * bsize

    # build joint distribution #
    table = np.empty((tsize, vsize*vsize + esize))
    idx = 0

    for sample in samples:

        sample = sample.reshape(sz, sz)

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

                table[idx, 0:vsize *
                      vsize] = np.reshape(sample[rl:ru, cl:cu], -1)
                for k in range(esize):
                    esamp = np.mod(np.array((r, c)) + env[k, :], sz)
                    table[idx, -(k+1)] = sample[esamp[0], esamp[1]]

                idx += 1
    table2 = np.apply_along_axis(row2bin, 1, table)

    # compute histogram
    thist = np.zeros((2**(vsize*vsize), 2**(esize)))

    for r, c in table2:
        thist[r, c] += 1

    thist /= tsize

    return thist


def row2bin(x):
    a = 0
    b = 0
    for i, j in enumerate(x):
        j = 1 if j == 1 else 0

        if i < vsize*vsize:
            a += j << i
        else:
            b += j << (i-vsize*vsize)
    return np.array([a, b])


# calculate and store if necessary #
####################################

try:
    joint_file = open(savefile, 'rb')
except IOError:
    print("Computing new joint distribution")
    thist = calculate_joint()
    joint_file = open(savefile, 'wb')
    np.save(joint_file, thist)
else:
    print("Loading saved joint distribution")
    thist = np.load(joint_file)

# information bottleneck test #
###############################

if perform_demo:
    dib = None

    dib = DIB(thist, beta=1, hiddens=50)
    dib.compress()
    dib.report_clusters()
    c = dib.visualize_clusters()

# beta sweep #
##############

if perform_beta_sweep:
    betas = np.linspace(0, 10, 21)
    info_y = np.zeros_like(betas)
    info_x = np.zeros_like(betas)

    for i, beta in enumerate(betas):
        dib = DIB(thist, beta=beta, hiddens=10)
        dib.compress()
        dib.report_clusters()
        info_y[i] = dib.mi_relevant()
        info_x[i] = dib.mi_captured()

    plt.plot(info_x, info_y, 'ko')
    plt.title('Information Plane Plot')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.show()
