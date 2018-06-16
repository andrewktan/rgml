import pickle

import matplotlib.pyplot as plt
import numpy as np


class DatasetIterator:
    def __init__(self, dfile, cgts=None, vsize=3, sz=81, debug=False):
        self.debug = debug

        self.nsamp = self.data.shape[0]

        self.vsize = vsize
        self.sz = sz

        self.idx = -1
        self.permutation = np.random.permutation(self.nsamp)

        if cgts == None:
            self.layers = 0
        else:
            self.layers = len(cgts)
        self.cgts = cgts

    def __iter__(self):
        return self

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

        return self.cgts[layer][a]
