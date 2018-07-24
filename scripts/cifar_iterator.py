import pickle

import matplotlib.pyplot as plt
import numpy as np


class CIFARIterator:
    """
    iterator for CIFAR-10 batch files
    """

    def __init__(self, dfile, img_size=32, num_channels=3, binarize=False):
        with open(dfile, 'rb') as fo:
            self.batch_dict = pickle.load(fo, encoding='bytes')
            self.data = self.batch_dict[b'data']

        self.nsamp = self.data.shape[0]

        self.imgshape = (num_channels, img_size, img_size)
        self.binarize = binarize

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
        item = item.transpose((1, 2, 0))

        if self.binarize:
            item = np.mean(item, axis=2)
            item = np.where(item > 255//2, 1, -1)

        return item


if __name__ == '__main__':
    dfile = '/Users/andrew/Documents/rgml/cifar-10_data/data_all'

    data = CIFARIterator(dfile, binarize=True)

    for idx, image in enumerate(data):
        if idx > 5:
            break

        if data.binarize:
            plt.imshow(image, cmap=plt.cm.gray)
        else:
            plt.imshow(image)

        plt.show()
