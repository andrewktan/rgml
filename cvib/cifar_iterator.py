import pickle

import matplotlib.pyplot as plt
import numpy as np

from dataset_iterator import *


class CIFARIterator(DatasetIterator):
    """
    iterator for CIFAR-10 batch files
    """

    def __init__(self, dfile, mb_size=32, vsz=3, grayscale=False, binarize=False, debug=False):

        # load all training files
        with open(dfile, 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
            self.data = batch_dict[b'data']
            self.labels = np.array(batch_dict[b'labels']).reshape(-1, 1)

        self.sz = 32
        self.grayscale = grayscale
        nchannels = 3

        if grayscale:
            # combine RGB channels
            nchannels = 1
            nsamp = self.data.shape[0]
            self.data = self.data.reshape((nsamp, 3, self.sz, self.sz))
            self.data = np.mean(self.data, axis=1) / 255
            self.data = self.data.reshape(nsamp, self.sz ** 2)

        self.binarize = binarize

        if binarize:
            nchannels = 1
            self.data = self.data > 0
            self.data = self.data.astype(np.int32)

        DatasetIterator.__init__(self, dfile, mb_size,
                                 vsz, nchannels, debug)

    def __next__(self):
        indices = np.random.randint(self.nsamp, size=self.mb_size)

        items = self.data[indices, :]

        if self.nchannels == 1:
            items = items.reshape(self.mb_size, self.sz**2)
        else:
            items = items.reshape(self.mb_size, self.nchannels, self.sz**2)
            items = items.transpose((0, 2, 1))

        labels = self.labels[indices]

        return items, labels


if __name__ == '__main__':
    dspath = '/Users/andrew/Documents/rgml/cifar-10_data/'
    dsname = 'data_all'
    dfile = "%s%s" % (dspath, dsname)

    it = CIFARIterator(dfile, grayscale=True, mb_size=32)
    samples, _ = it.next_batch()

    for idx in range(5):
        plt.imshow(np.reshape(samples[idx, :], (32, 32)), cmap=plt.cm.gray)
        plt.show()
