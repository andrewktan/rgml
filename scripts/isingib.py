import matplotlib.pyplot as plt
import numpy as np

from information_bottleneck import *
from ising_iterator import *

# parameters #
##############

perform_beta_sweep = False
perform_demo = True

dfile = '/Users/andrew/Documents/rgml/ising_data/data_0_45'
savefile = 'ising_ib_joint.npy'
sz = 81     # size of the samples (sq)
vsize = 3   # size of visible block (sq)
stride = 3
edist = 3  # size of buffer (sq)
esize = 4   # environment size
tsize = 1000000   # table size

# load data #
#############


def to_code(spins):
    ret = 0
    for i, j in enumerate(spins):
        j = 1 if j == 1 else 0
        ret += j << i

    return ret


def calculate_joint():
    samples = IsingIterator(dfile, img_size=sz)

    # choose environment #
    env = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * edist
    # env = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]]) * edist

    # build joint distribution #
    thist = np.zeros((2**(vsize*vsize), 2**(esize)))
    idx = 0

    for idx, sample in zip(range(tsize), samples):
        sample = sample.reshape(sz, sz)

        r, c = (sz//2, sz//2)   # sample from the middle

        rl = r - vsize//2
        ru = r + (vsize + 1)//2
        cl = c - vsize//2
        cu = c + (vsize + 1)//2

        vcode = to_code(np.reshape(sample[rl:ru, cl:cu], -1))

        esamps = np.empty(esize)

        for k in range(esize):
            esamp = np.array((r, c)) + env[k, :]
            esamps[k] = sample[esamp[0], esamp[1]]

        ecode = to_code(esamps)

        thist[vcode, ecode] += 1

    thist /= tsize

    return thist


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
    dib = DIB(thist, beta=15, hiddens=50)

    dib.compress()
    dib.report_clusters()
    c = dib.visualize_clusters(debug=True)

# beta sweep #
##############

if perform_beta_sweep:
    betas = np.arange(0, 30.1, 0.5)
    info_y = np.zeros_like(betas, dtype=np.float32)
    info_x = np.zeros_like(betas, dtype=np.float32)
    clusters = {x: [] for x in range(1, 100)}

    for i, beta in enumerate(betas):
        dib = DIB(thist, beta=beta, hiddens=100)
        dib.compress()
        f = dib.report_clusters()
        info_y[i] = dib.mi_relevant()
        info_x[i] = dib.mi_captured()
        clusters[np.unique(f).size].append(beta)

    # calculate kink angles
    angles = {k: np.pi/2 - np.arctan(np.min(v + [1e3])) -
              np.arctan(1/np.max(v + [1e-3]))
              for k, v in clusters.items()}
    angles = {k: v * 180/np.pi for k, v in angles.items()}
    angles = {k: max(v, 0) for k, v in angles.items()}

    plt.plot(info_x, info_y, 'ko')
    plt.title('DIB Plane Plot')
    plt.xlabel('H(T)')
    plt.ylabel('I(Y;T)')
    plt.show()

    with open("ipdata_%02d.pkl" % (2*edist+1), 'wb') as f:
        dump = {}
        dump['info_x'] = info_x
        dump['info_y'] = info_y
        dump['angles'] = angles
        dump['betas'] = betas
        dump['clusters'] = clusters
        dump['theta'] = np.array([angles[k] for k in clusters])
        pickle.dump(dump, f)
