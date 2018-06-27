import matplotlib.pyplot as plt
import numpy as np

from cifar_iterator import *
from information_bottleneck import *
from ising_iterator import *


class LayeredCoarseGrain:
    def __init__(self, dsname, dspath, Iterator, nl, sz=81, vsize=3, stride=3, env=None,
                 beta=1, hiddens=50, tsize=500000, debug=False):
        self.nl = nl            # number of layers
        self.dsname = dsname    # name of dataset
        self.fname = dspath + dsname   # full filepath of dataset
        self.Iterator = Iterator    # custom dataset iterator
        self.sz = sz            # system size
        self.vsize = vsize      # square size of visible block
        self.stride = stride    # stride in sampling
        self.debug = debug      # controls verbosity
        self.beta = beta        # Lagrange multiplier
        self.tsize = tsize      # max sample size
        self.hiddens = hiddens  # initialize number of clusters

        if env == None:         # custom environment, defaults to corners of 7x7
            self.env = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * 3
        else:
            self.env = env

        self.esize = self.env.shape[0]  # size of environment

        self.cur_layer = 0      # keep track of current layer
        self.pxys = []          # joint distributions of each layer
        self.dibs = []          # DIB objects from each layer
        self.cgts = []          # coarse graining transformations

    def run(self):
        """
        runs layered coarse graining procedure
        """
        for layer in range(self.nl):
            print("Starting layer %d" % layer)
            self._next_layer()

# helper functions #
    def _next_layer(self):
        """
        apply learned coarse graining to create next layer
        """
        self._get_joint()

        dib = DIB(self.pxys[self.cur_layer], beta=self.beta,
                  hiddens=self.hiddens)
        dib.compress(debug=self.debug)

        if self.debug:
            dib.report_clusters()

        self.dibs += [dib]
        self.cgts += [dib.f]
        self.cur_layer += 1

    def _get_joint(self):
        """
        get joint distribution of new layer
        """
        savefile = "%s_joint_l%d.npy" % (self.dsname, self.cur_layer)

        # load if possible
        try:
            joint_file = open(savefile, 'rb')
        except IOError:
            if self.debug:
                print("Computing new joint distribution")
            thist = self._build_joint()
            joint_file = open(savefile, 'wb')
            np.save(joint_file, thist)
        else:
            if self.debug:
                print("Loading saved joint distribution: %s" % savefile)
            thist = np.load(joint_file)

        self.pxys += [thist]

    def _build_joint(self):
        """
        calculate joint distribution of new layer
        """
        samples = self.Iterator(self.fname, cgts=self.cgts)

        # build joint distribution #
        table = np.empty((self.tsize, self.vsize**2 + self.esize))
        idx = 0

        for sample in samples:
            lsz = self.sz // (self.vsize ** self.cur_layer)

            sample = sample.reshape(lsz, lsz)

            for r in range(1, lsz, self.stride):
                for c in range(1, lsz, self.stride):
                    if idx >= self.tsize:
                        break

                    rl = np.mod(r - self.vsize//2, lsz)
                    ru = np.mod(r + (self.vsize + 1)//2, lsz)
                    cl = np.mod(c - self.vsize//2, lsz)
                    cu = np.mod(c + (self.vsize + 1)//2, lsz)

                    if rl > ru or cl > cu:  # hacky fix to wraparound
                        continue

                    table[idx, 0:self.vsize ** 2] = \
                        np.reshape(sample[rl:ru, cl:cu], -1)
                    for k in range(self.esize):
                        esamp = np.mod(np.array((r, c)) +
                                       self.env[k, :], lsz)
                        table[idx, -(k+1)] = sample[esamp[0], esamp[1]]

                    idx += 1

        table2 = np.apply_along_axis(self._row2bin, 1, table)

        # compute histogram
        thist = np.zeros((2**(self.vsize**2), 2**(self.esize)))

        for r, c in table2:
            thist[r, c] += 1

        # resize
        thist = thist[0:idx, :]
        thist /= table2.shape[0]

        # debug info
        if self.debug:
            print("Number of samples: %d" % table2.shape[0])

        return thist

    def _row2bin(self, x):
        a = 0
        b = 0
        for i, j in enumerate(x):
            j = 1 if j == 1 else 0

            if i < self.vsize**2:
                a += j << i
            else:
                b += j << (i-self.vsize**2)
        return np.array([a, b])

# reporting and visualization #
    def get_ib_object(self, k):
        """
        returns IB object from k-th layer
        """
        return self.dibs[k]

    def visualize_clusters(self):
        """
        display clusters from all layers
        """
        for layer in range(self.nl):
            self.dibs[layer].visualize_clusters(debug=True)


if __name__ == '__main__':

    demo_ising = True
    demo_cifar = False

    if demo_ising:
        dspath = '/Users/andrew/Documents/rgml/ising_data/'
        dsname = 'data_0_45'
        dfile = "%s%s" % (dspath, dsname)

        model = LayeredCoarseGrain(
            dsname, dspath, IsingIterator, 3, beta=3, sz=81, debug=True)
        model.run()

        model.visualize_clusters()

    if demo_cifar:
        dspath = '/Users/andrew/Documents/rgml/cifar-10_data/'
        dsname = 'data_batch_2'
        dfile = "%s%s" % (dspath, dsname)

        model = LayeredCoarseGrain(
            dsname, dspath, CIFARIterator, 2, beta=4, sz=32, debug=True)
        model.run()

        model.visualize_clusters()
