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

from ising_iterator import *


class JointDistribution:
    def __init__(self, dsname, dspath, sz=81, vsz=3, tsz=1e6, stride=3,
                 env=None, debug=False):
        self.dsname = dsname    # name of dataset
        self.fname = dspath + dsname   # full filepath of dataset
        self.sz = sz            # system size
        self.vsz = vsz          # square size of visible block
        self.stride = stride    # stride in sampling
        self.debug = debug      # controls verbosity

        if env == None:         # custom environment, defaults to corners of 5x5
            self.env = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * 2
        else:
            self.env = env

        self.esz = self.env.shape[0]  # size of environment

    def _get_samples(self):
        """
        try to load saved or build new
        """
        savefile = "%s_samples.npy" % self.dsname

        # load if possible
        try:
            joint_file = open(savefile, 'rb')
        except IOError:
            if self.debug:
                print("Computing new joint distribution")
            thist = self._build_joint()
            joint_file = open(savefile, 'wb')
            np.save(joint_file, thist)
        else:
            if self.debug:
                print("Loading saved joint distribution: %s" % savefile)
            thist = np.load(joint_file)

        self.pxys += [thist]

    def _build_samples(self):
        """
        build samples
        """
        samples = self.Iterator(self.fname, cgts=self.cgts)

        # build joint distribution #
        table = np.empty((self.tsz, self.vsz**2 + self.esz))
        idx = 0

        for sample in samples:
            lsz = self.sz // (self.vsz ** self.cur_layer)

            sample = sample.reshape(lsz, lsz)

            for r in range(1, lsz, self.stride):
                for c in range(1, lsz, self.stride):
                    if idx >= self.tsz:
                        break

                    rl = np.mod(r - self.vsz//2, lsz)
                    ru = np.mod(r + (self.vsz + 1)//2, lsz)
                    cl = np.mod(c - self.vsz//2, lsz)
                    cu = np.mod(c + (self.vsz + 1)//2, lsz)

                    if rl > ru or cl > cu:  # hacky fix to wraparound
                        continue

                    table[idx, 0:self.vsz ** 2] = \
                        np.reshape(sample[rl:ru, cl:cu], -1)
                    for k in range(self.esz):
                        esamp = np.mod(np.array((r, c)) +
                                       self.env[k, :], lsz)
                        table[idx, -(k+1)] = sample[esamp[0], esamp[1]]

                    idx += 1

        # debug info
        if self.debug:
            print("Number of samples: %d" % table2.shape[0])

        return table


if False:
    mb_size = 32
    Z_dim = 100 * 2
    h_dim = 128
    c = 0
    lr = 1e-3

    if True:
        dspath = '/Users/andrew/Documents/rgml/cifar-10_data/'
        dsname = 'data_batch_2'
        dfile = "%s%s" % (dspath, dsname)
        dset = CIFARIterator(dfile, mb_size=mb_size)

    if False:
        dspath = '/Users/andrew/documents/rgml/ising_data/'
        dsname = 'data_0_50'
        dfile = "%s%s" % (dspath, dsname)
        dset = IsingIterator(dfile, mb_size=mb_size)

    sz = int(dset.sz)
    X_dim = sz**2

    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

    # =============================== Q(z|X) ======================================

    Wxh = xavier_init(size=[X_dim, h_dim])
    bxh = Variable(torch.zeros(h_dim), requires_grad=True)

    Whz_mu = xavier_init(size=[h_dim, Z_dim])
    bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

    Whz_var = xavier_init(size=[h_dim, Z_dim])
    bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)

    def Q(X):
        h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
        z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
        z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
        return z_mu, z_var

    def sample_z(mu, log_var):
        eps = Variable(torch.randn(mb_size, Z_dim))
        return mu + torch.exp(log_var / 2) * eps

    # =============================== P(X|z) ======================================

    Wzh = xavier_init(size=[Z_dim, h_dim])
    bzh = Variable(torch.zeros(h_dim), requires_grad=True)

    Whx = xavier_init(size=[h_dim, X_dim])
    bhx = Variable(torch.zeros(X_dim), requires_grad=True)

    def P(z):
        h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
        X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
        return X

    # =============================== TRAINING ====================================

    params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
              Wzh, bzh, Whx, bhx]

    solver = optim.Adam(params, lr=lr)

    for it, X in zip(range(100000), dset):
        X = Variable(torch.from_numpy(X)).float()

        # Forward
        z_mu, z_var = Q(X)
        z = sample_z(z_mu, z_var)
        X_sample = P(z)

        # Loss
        recon_loss = nn.binary_cross_entropy(
            X_sample, X, size_average=False) / mb_size
        kl_loss = torch.mean(
            0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        loss = recon_loss + kl_loss

        # Backward
        loss.backward()

        # Update
        solver.step()

        # Housekeeping
        for p in params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resz_as_(data).zero_())

        # Print and plot every now and then
        if it % 1000 == 0:
            print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

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
