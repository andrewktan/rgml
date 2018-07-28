import pickle

import matplotlib.pyplot as plt
import numpy as np


class IsingIterator:
    """
    iterator for Ising batch files
    """

    def __init__(self, dfile, img_size=25, roll=True, seed=0):
        with open(dfile, 'rb') as fo:
            self.data = np.loadtxt(fo, dtype=np.int32)

        self.nsamp = self.data.shape[0]

        self.imgshape = (img_size, img_size)

        self.roll = roll    # translational invariance on torus

        np.random.seed(seed)

        if not roll:
            self.idx = -1
            self.permutation = np.random.permutation(self.nsamp)

    def __iter__(self):
        return self

    def __next__(self):
        if self.roll:
            idx = np.random.randint(self.nsamp)
            r, c = np.random.randint(self.imgshape[0], size=2)

            item = self.data[idx, :]
            item = item.reshape(self.imgshape)

            item = np.roll(item, r, axis=0)
            item = np.roll(item, c, axis=1)
        else:
            self.idx += 1
            if self.idx >= self.nsamp:
                raise StopIteration

            item = self.data[self.idx, :]
            item = item.reshape(self.imgshape)

        return item


if __name__ == '__main__':
    dfile = '/Users/andrew/Documents/rgml/cifar-10_data/data_all_cg_03_15_full'

    data = IsingIterator(dfile, img_size=30, roll=False)

    for idx, image in enumerate(data):
        if idx > 5:
            break

        plt.imshow(image, cmap=plt.cm.gray)

        plt.show()
