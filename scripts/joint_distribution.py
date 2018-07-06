import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.autograd as autograd
import torch.nn.functional as nn
import torch.optim as optim
from torch.autograd import Variable

from mnist_iterator import *


class JointDistribution:
    def __init__(self, dset, mb_size=32, Z_dim=10, h_dim=128,
                 lr=1e-3, debug=False):

        self.debug = debug      # controls verbosity

        # VAE parameters
        self.mb_size = mb_size       # mini-batch size
        self.Z_dim = Z_dim         # compressed size
        self.h_dim = h_dim        # hidden layer size
        self.lr = lr          # learning rate

        sz = int(dset.sz)
        X_dim = sz**2

        # decoder
        self.Wzh = self.xavier_init(size=[Z_dim, h_dim])
        self.bzh = Variable(torch.zeros(h_dim), requires_grad=True)

        self.Whx = self.xavier_init(size=[h_dim, X_dim])
        self.bhx = Variable(torch.zeros(X_dim), requires_grad=True)

        # encoder
        self.Wxh = self.xavier_init(size=[X_dim, h_dim])
        self.bxh = Variable(torch.zeros(h_dim), requires_grad=True)

        self.Whz_mu = self.xavier_init(size=[h_dim, Z_dim])
        self.bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

        self.Whz_var = self.xavier_init(size=[h_dim, Z_dim])
        self.bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

    def encoder(self, X):
        h = nn.relu(X @ self.Wxh + self.bxh.repeat(X.size(0), 1))
        z_mu = h @ self.Whz_mu + self.bhz_mu.repeat(h.size(0), 1)
        z_var = h @ self.Whz_var + self.bhz_var.repeat(h.size(0), 1)
        return z_mu, z_var

    def sample_z(self, mu, log_var):
        eps = Variable(torch.randn(self.mb_size, self.Z_dim))
        return mu + torch.exp(log_var / 2) * eps

    def decoder(self, z):
        h = nn.relu(z @ self.Wzh + self.bzh.repeat(z.size(0), 1))
        X = nn.sigmoid(h @ self.Whx + self.bhx.repeat(h.size(0), 1))
        return X

    def train(self):
        c = 0

        params = [self.Wxh, self.bxh, self.Whz_mu, self.bhz_mu, self.Whz_var,
                  self.bhz_var, self.Wzh, self.bzh, self.Whx, self.bhx]

        solver = optim.Adam(params, lr=self.lr)

        for it, X in zip(range(10000), dset):
            X = Variable(torch.from_numpy(X)).float()

            # forward
            z_mu, z_var = self.encoder(X)
            z = self.sample_z(z_mu, z_var)
            X_sample = self.decoder(z)

            # loss
            recon_loss = nn.binary_cross_entropy(
                X_sample, X, size_average=False) / self.mb_size
            kl_loss = torch.mean(
                0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
            loss = recon_loss + kl_loss

            # backward
            loss.backward()

            # update
            solver.step()

            # housekeeping
            for p in params:
                if p.grad is not None:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())

            # print and plot
            if it % 1000 == 0:
                print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

                if self.debug:
                    samples = P(z).data.numpy()[:16]

                    fig = plt.figure(figsize=(4, 4))
                    gs = gridspec.GridSpec(4, 4)
                    gs.update(wspace=0.05, hspace=0.05)

                    for i, sample in enumerate(samples):
                        ax = plt.subplot(gs[i])
                        plt.axis('off')
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        ax.set_aspect('equal')
                        plt.imshow(sample.reshape(sz, sz), cmap='Greys_r')

                    if not os.path.exists('out/'):
                        os.makedirs('out/')

                    plt.savefig('out/{}.png'.format(str(c).zfill(3)),
                                bbox_inches='tight')
                    c += 1
                    plt.close(fig)


if __name__ == '__main__':
    dspath = '/Users/andrew/documents/rgml/mnist_data/'
    dsname = 'train-images-idx3-ubyte.gz'
    dfile = "%s%s" % (dspath, dsname)
    dset = MNISTIterator(dfile, mb_size=32)

    pxy = JointDistribution(dset)
    pxy.train()
