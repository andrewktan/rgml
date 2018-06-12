import pickle

import matplotlib.pyplot as plt
import numpy as np


class IsingIterator:
    """
    iterator for Ising batch files
    """

    def __init__(self, dfile, img_size=25):
        with open(dfile, 'rb') as fo:
            self.data = np.loadtxt(fo, dtype=np.int32)

        self.nsamp = self.data.shape[0]

        self.imgshape = (img_size, img_size)

        self.idx = -1
        self.permutation = np.random.permutation(self.nsamp)

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= self.nsamp:
            raise StopIteration

        item = self.data[self.idx, :]
        item = item.reshape(self.imgshape)

        return item


if __name__ == '__main__':
    dfile = '/Users/andrew/Documents/rgml/ising/data_0_50.txt'
#    dfile = '/Users/andrew/rgml/ising/data.txt'

    data = IsingIterator(dfile)

    for idx, image in enumerate(data):
        if idx > 5:
            break

        plt.imshow(image, cmap=plt.cm.gray)

        plt.show()
