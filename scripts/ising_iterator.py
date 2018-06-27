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

    def __init__(self, dfile, mb_size=1, vsz=3, debug=False):
        with open(dfile, 'rb') as fo:
            self.data = np.loadtxt(fo)
            self.data = self.data > 0
            self.data = self.data.astype(np.int32)

        DatasetIterator.__init__(self, dfile, mb_size, vsz, debug)

    def __next__(self):

        indicies = np.random.randint(self.nsamp, size=self.mb_size)

        items = self.data[indicies, :]
        items = items.reshape(self.mb_size, self.sz ** 2)

        return items


if __name__ == '__main__':
    dspath = '/Users/andrew/documents/rgml/ising_data/'
    dsname = 'data_0_50'
    dfile = "%s%s" % (dspath, dsname)

    it = IsingIterator(dfile, mb_size=32)
    samples = it.__next__()

    plt.imshow(np.reshape(samples[0, :], (25, 25)), cmap=plt.cm.gray)
    plt.show()
