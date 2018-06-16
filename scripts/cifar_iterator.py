import pickle

import matplotlib.pyplot as plt
import numpy as np
from dataset_iterator import *
from layered_coarse_grain import *


class CIFARIterator(DatasetIterator):
    """
    iterator for CIFAR-10 batch files
    """

    def __init__(self, dfile, cgts=None, vsize=3, sz=32, num_channels=3,
                 binarize=True, debug=False):
        with open(dfile, 'rb') as fo:
            self.batch_dict = pickle.load(fo, encoding='bytes')
            self.data = self.batch_dict[b'data']

        DatasetIterator.__init__(self, dfile, cgts, vsize, sz, debug)

        self.imgshape = (num_channels, sz, sz)
        self.binarize = binarize

    def __next__(self):
        self.idx += 1
        if self.idx >= self.nsamp:
            raise StopIteration

        sample = self.data[self.idx, :]
        sample = sample.reshape(self.imgshape)
        sample = sample.transpose((1, 2, 0))

        if self.binarize:
            sample = np.mean(sample, axis=2)
            sample = np.where(sample > 255//2, 1, -1)

        sample = self._coarsegrain(sample)

        return sample


if __name__ == '__main__':
    dspath = '/Users/andrew/Documents/rgml/cifar-10_data/'
    dsname = 'data_batch_2'
    dfile = "%s%s" % (dspath, dsname)

    model = LayeredCoarseGrain(
        dsname, dspath, CIFARIterator, 1, sz=32, beta=30, debug=True)
    model.run()

    f = model.get_ib_object(0).f

    it = CIFARIterator(dfile, cgts=[f])

    for idx, sample in enumerate(it):
        if idx > 5:
            break
        plt.matshow(sample, cmap=plt.cm.gray)
        plt.show()
