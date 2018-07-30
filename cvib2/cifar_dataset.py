import pickle

import matplotlib.pyplot as plt
import numpy as np


class CIFARDataset():
    """
    dataset for binarized CIFAR-10 batch files
    """

    def __init__(self,
                 dfile,
                 sz=32,
                 mb_size=32,
                 grayscale=False,
                 binarize=False,
                 debug=False):

        # load all training files
        with open(dfile, 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
            self.data = batch_dict[b'data']
            self.labels = np.array(batch_dict[b'labels']).reshape(-1, 1)

        self.sz = sz
        self.mb_size = mb_size
        self.grayscale = grayscale or binarize
        self.binarize = binarize
        self.nsamp = self.data.shape[0]

        if mb_size == -1:
            self.mb_size = self.nsamp

        if self.grayscale:
            self.data = self.data.reshape(self.nsamp, 3, sz, sz)
            self.data = np.mean(self.data, axis=1) / 255
            self.data = self.data.reshape(self.nsamp, self.sz ** 2)

        # binarize
        if binarize:
            self.data = self.data > 0.5
            self.data = self.data.astype(np.int32)

    def set_mbsize(self, mb_size):
        self.mb_size = mb_size

    def __next__(self):
        indices = np.random.randint(self.nsamp, size=self.mb_size)

        items = self.data[indices, :]
        items = items.reshape(self.mb_size, self.sz**2)

        labels = self.labels[indices]

        return items, labels

    def next_batch(self):
        return self.__next__()


if __name__ == '__main__':
    dspath = '/Users/andrew/Documents/rgml/cifar-10_data/'
    dsname = 'data_all'
    dfile = "%s%s" % (dspath, dsname)

    sz = 32

    it = CIFARDataset(dfile, sz=sz, grayscale=True)
    samples, _ = it.next_batch()

    for idx in range(5):
        plt.imshow(np.reshape(samples[idx, :], (sz, sz)), cmap=plt.cm.gray)
        plt.show()
