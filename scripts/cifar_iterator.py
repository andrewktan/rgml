import pickle

import matplotlib.pyplot as plt
import numpy as np

from dataset_iterator import *
from layered_coarse_grain import *


class CIFARIterator(DatasetIterator):
    """
    iterator for CIFAR-10 batch files
    """

    def __init__(self, dfile, mb_size=32, vsz=3, binarize=False, debug=False):
        with open(dfile, 'rb') as fo:
            self.batch_dict = pickle.load(fo, encoding='bytes')
            self.data = self.batch_dict[b'data']

            # combine RGB channels
            self.sz = 32
            self.nsamp = self.data.shape[0]
            self.data = self.data.reshape((self.nsamp, 3, self.sz, self.sz))
            self.data = np.mean(self.data, axis=1) / 255
            self.data = self.data.reshape(self.nsamp, self.sz ** 2)

        self.binarize = binarize

        if binarize:
            self.data = self.data > 0
            self.data = self.data.astype(np.int32)

        DatasetIterator.__init__(self, dfile, mb_size, vsz, debug)


if __name__ == '__main__':
    dspath = '/Users/andrew/Documents/rgml/cifar-10_data/'
    dsname = 'data_batch_2'
    dfile = "%s%s" % (dspath, dsname)

    it = CIFARIterator(dfile, mb_size=32)
    samples = it.__next__()

    plt.imshow(np.reshape(samples[0, :], (32, 32)), cmap=plt.cm.gray)
    plt.show()
