import gzip

import matplotlib.pyplot as plt
import numpy as np

from dataset_iterator import *
from layered_coarse_grain import *


class MNISTIterator(DatasetIterator):
    """
    iterator for MNIST files
        dfile - path to data file
    """

    def __init__(self, dfile, mb_size=1, vsz=3, debug=False):
        with gzip.open(dfile) as f:
            self.data = np.frombuffer(f.read(), 'B', offset=16)
            self.data = np.reshape(self.data, (-1, 784))
            self.data = self.data / self.data.max()

        DatasetIterator.__init__(self, dfile, mb_size, vsz, debug)


if __name__ == '__main__':
    dspath = '/Users/andrew/documents/rgml/mnist_data/'
    dsname = 'train-images-idx3-ubyte.gz'
    dfile = "%s%s" % (dspath, dsname)

    it = MNISTIterator(dfile, mb_size=32)
    samples = it.__next__()

    for k in range(5):
        plt.imshow(np.reshape(samples[k, :], (28, 28)), cmap=plt.cm.gray)
        plt.show()
