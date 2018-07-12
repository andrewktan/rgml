import matplotlib.pyplot as plt
import numpy as np


class DatasetIterator:
    def __init__(self, dfile, mb_size=1, vsz=3, nchannels=1, debug=False):
        self.debug = debug

        self.nsamp = self.data.shape[0]  # number of samples

        # image dimension
        self.sz = np.sqrt(self.data.shape[1]/nchannels).astype(int)
        self.vsz = vsz      # size of visible patch
        self.nchannels = nchannels

        self.num_samples = self.data.shape[0]

        if mb_size == -1:
            mb_size = self.num_samples

        self.mb_size = mb_size  # mini batch size

    def __iter__(self):
        return self

    def __next__(self):
        indices = np.random.randint(self.nsamp, size=self.mb_size)

        items = self.data[indices, :]

        if self.nchannels > 1:
            items = items.reshape(self.mb_size, self.sz**2, self.nchannels)
        else:
            items = items.reshape(self.mb_size, self.sz**2)

        labels = self.labels[indices]

        return items, labels

    def next_batch(self):
        return self.__next__()
