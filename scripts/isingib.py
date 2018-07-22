import matplotlib.pyplot as plt
import numpy as np

from information_bottleneck import *
from ising_iterator import *

# parameters #
##############

perform_beta_sweep = False
perform_demo = True

symmetrize = True

sz = 81     # size of the samples (sq)
vsz = 3   # size of visible block (sq)
edist = 9  # size of buffer (sq)
esz = 4   # environment size
tsz = 1000000   # table size
dfile = '/Users/andrew/Documents/rgml/ising_data/data_0_45'
savefile = "/Users/andrew/Documents/rgml/ip_data/vanilla2/isingib_joint_%02d.npy" % (
    2*edist+1)
# savefile = "/Users/andrew/Documents/rgml/ip_data/strawberry/isingib_q_joint_%02d.npy" % edist

# load data #
#############

def to_code(spins):
    ret = 0
    for i, j in enumerate(spins):
        j = 1 if j == 1 else 0
        ret += j << i

    return ret


def calculate_joint(edist):
    samples = IsingIterator(dfile, img_size=sz)

    # choose environment #
    env = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * edist
    # env = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]]) * edist

    # build joint distribution #
    thist = np.zeros((2**(vsz*vsz), 2**(esz)))
    idx = 0

    for idx, sample in zip(range(tsz), samples):
        sample = sample.reshape(sz, sz)

        r, c = (sz//2, sz//2)   # sample from the middle

        rl = r - vsz//2
        ru = r + (vsz + 1)//2
        cl = c - vsz//2
        cu = c + (vsz + 1)//2

        vcode = to_code(np.reshape(sample[rl:ru, cl:cu], -1))

        esamps = np.empty(esz)

        for k in range(esz):
            esamp = np.array((r, c)) + env[k, :]
            esamps[k] = sample[esamp[0], esamp[1]]

        ecode = to_code(esamps)

        thist[vcode, ecode] += 1

    thist /= tsz

    return thist


# calculate and store if necessary #
####################################

try:
    joint_file = open(savefile, 'rb')
except IOError:
    print("Computing new joint distribution")
    thist = calculate_joint(edist)
    joint_file = open(savefile, 'wb')
    np.save(joint_file, thist)
    print("Saved new joint distribution as %s" % savefile)
else:
    print("Loading saved joint distribution %s" % savefile)
    thist = np.load(joint_file)

# information bottleneck test #
###############################

if symmetrize and (thist.shape[0] == thist.shape[1]):
    thist = (thist + thist.T)/2

if perform_demo:
    beta = 100
    dib = DIB(thist, beta=beta, hiddens=100)

    dib.compress()
    c = dib.visualize_clusters(debug=True)

    # save clustering
    with open("out/cg_%02d_%02d.pkl" % (2*edist+1, beta), 'wb') as f:
        dump = {}
        dump['f'] = dib.report_clusters()
        pickle.dump(dump, f)

# beta sweep #
##############

if perform_beta_sweep:
    betas = np.arange(0, 30.1, 0.5)
    hiddens = 100
    info_y = np.zeros_like(betas, dtype=np.float32)
    info_x = np.zeros_like(betas, dtype=np.float32)
    clusters = {x: [] for x in range(1, hiddens)}
    clusters2 = np.zeros_like(betas, dtype=np.unit8)
    clusterings = {}

    for i, beta in enumerate(betas):
        dib = DIB(thist, beta=beta, hiddens=hiddens)
        dib.compress()
        f = dib.report_clusters()
        info_y[i] = dib.mi_relevant()
        info_x[i] = dib.mi_captured()
        clusters[np.unique(f).size].append(beta)
        clusters2[i] = np.unique(f).size
        clusterings[beta] = dib.f

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

    with open("out/ipdata_%02d.pkl" % (2*edist+1), 'wb') as f:
        dump = {}
        dump['info_x'] = info_x
        dump['info_y'] = info_y
        dump['angles'] = angles
        dump['betas'] = betas
        dump['clusters'] = clusters
        dump['clusters2'] = clusters2
        dump['clusterings'] = clusterings
        dump['theta'] = np.array([angles[k] for k in clusters])
        pickle.dump(dump, f)
