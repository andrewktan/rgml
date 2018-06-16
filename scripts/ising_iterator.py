import pickle

import matplotlib.pyplot as plt
import numpy as np
from layered_coarse_grain import *


class IsingIterator:
    """
    iterator for Ising batch files
        dfile - path to .txt data file
        cgt - list of coarsegraining maps
        sz - system size
    """

    def __init__(self, dfile, cgt=None, vsize=3, sz=25, debug=False):
        with open(dfile, 'rb') as fo:
            self.data = np.loadtxt(fo, dtype=np.int32)

        self.debug = debug

        self.nsamp = self.data.shape[0]

        self.vsize = vsize
        self.sz = sz

        self.idx = -1
        self.permutation = np.random.permutation(self.nsamp)

        if cgt == None:
            self.layers = 0
        else:
            self.layers = len(cgt)
        self.cgt = cgt

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= self.nsamp:
            raise StopIteration

        item = self.data[self.idx, :]
        item = item.reshape(self.sz, self.sz)
        item = self._coarsegrain(item)

        return item

    def _coarsegrain(self, sample):
        """
        perform coarse graining procedure using clusters
        """
        # apply coarse graining
        sz = self.sz
        cg_sample = sample

        for layer in range(self.layers):
            sz = sz//self.vsize
            cg_sample = np.empty((sz, sz))
            for r in range(sz):
                for c in range(sz):
                    patch = sample[r*self.vsize:(r+1)*self.vsize,
                                   c*self.vsize:(c+1)*self.vsize]
                    patch = patch.reshape(-1)

                    cg_sample[r, c] = self._transform_patch(patch, layer)

            sample = cg_sample

        return sample

    def _transform_patch(self, x, layer):
        a = 0
        for i, j in enumerate(x):
            j = 1 if j == 1 else 0
            a += j << i

        return self.cgt[layer][a]


if __name__ == '__main__':
    dspath = '/Users/andrew/Documents/rgml/ising_data/'
    dsname = 'data_0_50'
    dfile = "%s%s.txt" % (dspath, dsname)

    model = LayeredCoarseGrain(dsname, dspath, 1, beta=1, debug=True)
    model.run()

    f = model.get_ib_object(0).f

    it = IsingIterator(dfile, cgt=[f, f])

    for idx, sample in enumerate(it):
        if idx > 5:
            break
        plt.matshow(sample, cmap=plt.cm.gray)
        plt.show()
