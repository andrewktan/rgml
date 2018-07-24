import matplotlib.pyplot as plt
import numpy as np

from cifar_iterator import *
from information_bottleneck_q import *

# parameters #
##############

perform_beta_sweep = False
perform_demo = True

symmetrize = True

sz = 32     # size of the samples (sq)
vsz = 3   # size of visible block (sq)
edist = 3   # distance to environment patch
stride = 8
tsz = 1000000   # table size
dfile = '/Users/andrew/Documents/rgml/cifar-10_data/data_all'
savefile = "out/cifarib_q_joint_%02d.npy" % edist

# load data #
#############


def to_code(spins):
    ret = 0
    for i, j in enumerate(spins):
        j = 1 if j == 1 else 0
        ret += j << i

    return ret


def calculate_joint(eloc):
    """
    Calculates the joint p(x,x')
    """
    thist = np.zeros((2**(vsz**2), 2**(vsz**2)))
    idx = 0

    for sample in CIFARIterator(dfile, binarize=True):
        for r in range(0, sz, stride):
            for c in range(0, sz, stride):
                if idx >= tsz:
                    break

                # careful about wraparound
                rl = np.mod(r - vsz//2, sz)
                ru = np.mod(r + (vsz + 1)//2, sz)
                cl = np.mod(c - vsz//2, sz)
                cu = np.mod(c + (vsz + 1)//2, sz)

                vcode = to_code(np.reshape(sample[rl:ru, cl:cu], -1))
                ecode = to_code(np.reshape(
                    sample[rl+eloc:ru+eloc, cl:cu], -1))

                thist[vcode, ecode] += 1
                idx += 1

    if idx != tsz:
        print("Insuffcient number of samples (%d)" % idx)

    thist /= idx

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
if symmetrize:
    thist = (thist + thist.T)/2

if perform_demo:
    dib = DIB(thist, beta=15, hiddens=30)
    dib.compress()
    dib.report_clusters()
    c = dib.visualize_clusters(debug=True)

# beta sweep #
##############

if perform_beta_sweep:
    betas = np.arange(0, 80.1, 1.0)
    hiddens = 100
    info_y = np.zeros_like(betas, dtype=np.float32)
    info_x = np.zeros_like(betas, dtype=np.float32)
    clusters = {x: [] for x in range(1, hiddens)}
    clusters2 = np.zeros_like(betas, dtype=np.uint8)
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

    with open("out/ipdata_cq_%02d.pkl" % edist, 'wb') as f:
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
