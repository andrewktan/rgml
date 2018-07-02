import pickle

import matplotlib.pyplot as plt
import numpy as np

from information_bottleneck import *
from ising_iterator import *

# parameters #
##############

perform_beta_sweep = False
perform_demo = True

dfile = '/Users/andrew/Documents/rgml/ising_data/data_0_50'
savefile = 'ising_ib_table.pkl'
sz = 25     # size of the samples (sq)
vsz = 3   # size of visible block (sq)
stride = 3
bsz = 7   # size of buffer (sq)
tsize = 10000   # table size

# load data #
#############


def rotate_torus(A, r0, c0):
    """
    use period boundaries to circulate rows and columns
        places specified (r0, c0) element at (0, 0) position
    """
    sr, sc = A.shape

    ret = A.take(range(r0, r0+sr), mode='wrap', axis=0)
    ret = ret.take(range(c0, c0+sc), mode='wrap', axis=0)

    return ret


def generate_samples():
    samples = IsingIterator(dfile)

    # build joint distribution #
    X = np.empty((tsize, vsz**2), dtype=np.int8)
    Y = np.empty((tsize, sz**2 - bsz**2), dtype=np.int8)
    idx = 0

    for sample in samples:

        sample = sample.reshape(sz, sz)

        for r in range(1, sz, stride):
            for c in range(1, sz, stride):
                if idx >= tsize:
                    break

                # calculate indices
                vr = np.mod(r - vsz//2, sz)
                vc = np.mod(c - vsz//2, sz)
                br = np.mod(c - bsz//2, sz)
                bc = np.mod(c - bsz//2, sz)

                # capture visible patch
                sample_rot = rotate_torus(sample, vr, vc)

                X[idx, :] = np.reshape(sample_rot[0:vsz, 0:vsz], -1)

                # capture environment
                sample_rot = rotate_torus(sample, br, bc)
                sample_rot[0:bsz, 0:bsz] = 0
                sample_rot = np.reshape(sample_rot, -1)
                Y[idx, :] = sample_rot[sample_rot != 0]

                idx += 1

    X = X[0:idx, :]
    Y = Y[0:idx, :]

    return X, Y


# calculate and store if necessary #
####################################

try:
    joint_file = open(savefile, 'rb')
except IOError:
    print("Computing new joint distribution")
    X, Y = generate_samples()
    joint_file = open(savefile, 'wb')
    pickle.dump([X, Y], joint_file)
else:
    print("Loading saved joint distribution")
    X, Y = pickle.load(joint_file)

# information bottleneck test #
###############################

if perform_demo:
    dib = None

    dib = DIB(X, Y)
    dib.compress()
    dib.report_clusters()
    c = dib.visualize_clusters(debug=True)

# beta sweep #
##############

if perform_beta_sweep:
    betas = np.arange(0, 6.05, 0.1)
    info_y = np.zeros_like(betas)
    info_x = np.zeros_like(betas)
    clusters = {x: [] for x in range(1, 20)}

    for i, beta in enumerate(betas):
        dib = DIB(thist, beta=beta, hiddens=50)
        dib.compress()
        f = dib.report_clusters()
        info_y[i] = dib.mi_relevant()
        info_x[i] = dib.mi_captured()
        clusters[np.unique(f).size].append(beta)

    # calculate kink angles
    angles = {k: np.pi/2 - np.arctan(np.min(v + [100])) -
              np.arctan(1/np.max(v + [0]))
              for k, v in clusters.items()}
    angles = {k: v * 180/np.pi for k, v in angles.items()}
    angles = {k: max(v, 0) for k, v in angles.items()}

    plt.plot(info_x, info_y, 'ko')
    plt.title('Information Plane Plot')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.show()
