import pickle

import matplotlib.pyplot as plt
import numpy as np
from dataset_iterator import *
from layered_coarse_grain import *


class IsingIterator(DatasetIterator):
    """
    iterator for Ising batch files
        dfile - path to data file
        cgts - list of coarsegraining maps
        sz - system size
    """

    def __init__(self, dfile, cgts=None, vsize=3, sz=81, debug=False):
        with open(dfile, 'rb') as fo:
            self.data = np.loadtxt(fo, dtype=np.int32)

        DatasetIterator.__init__(self, dfile, cgts, vsize, sz, debug)

    def __next__(self):
        self.idx += 1
        if self.idx >= self.nsamp:
            raise StopIteration

        item = self.data[self.idx, :]
        item = item.reshape(self.sz, self.sz)
        item = self._coarsegrain(item)

        return item


if __name__ == '__main__':
    dspath = '/Users/andrew/Documents/rgml/ising_data/'
    dsname = 'data_0_45'
    dfile = "%s%s" % (dspath, dsname)

    model = LayeredCoarseGrain(dsname, dspath, 1, beta=1, debug=True)
    model.run()

    f = model.get_ib_object(0).f

    it = IsingIterator(dfile, cgts=[f, f])

    for idx, sample in enumerate(it):
        if idx > 5:
            break
        plt.matshow(sample, cmap=plt.cm.gray)
        plt.show()
