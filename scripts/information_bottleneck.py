from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma


class DIB:
    eps = 1e-12

    def __init__(self, X, Y, beta=5):
        self.beta = beta

        self.X = X
        self.Y = Y

        self.X_uniques = np.vstack({tuple(x) for x in X})
        self.clusters = len(self.X_uniques)

        # cluster assignments
        self.f = {tuple(v): i for i, v in enumerate(self.X_uniques)}

    def compress(self):
        """
        optimize DIB cost function
        """
        self._greedy_merge()

# core DIB helper functions #
    def _greedy_merge(self):
        """
        greedly merge two clusters
        """
        best_cost = 1e6
        best_a, best_b = None, None

        for a, b in combinations(range(self.clusters), 2):
            print("trying %d %d" % (a, b))
            print("best cost: %.3f" % best_cost)
            a = tuple(self.X_uniques[a])
            b = tuple(self.X_uniques[b])

            cost = self._dib_cost(a, b)
            print("cost: %.3f" % cost)

            if cost < best_cost:
                best_cost = cost
                best_a, best_b = a, b

    def _dib_cost(self, a, b):
        """
        calculate DIB cost function
        """
        T = np.array([[self.f[tuple(k)] if tuple(k) !=
                       a else self.f[b] for k in self.X]]).T

        cost = self._mi_estimator(self.X, self.Y)
        cost -= self.beta * self._mi_estimator(self.X, T)

        return cost

    def _x_norm(self, X):
        """
        norm on visible patch space
        """
        return np.linalg.norm(X, ord=1, axis=-1)

    def _y_norm(self, Y):
        """
        metric on environment space
        """
        return np.linalg.norm(Y, ord=1, axis=-1)

    def _mi_estimator(self, X, Y, k=5):
        """
        first estimator for mutual information described by Kraskov et al. (2008)
        """
        N = X.shape[0]

        # calculate estimator
        I = digamma(k) + digamma(N)

        for n, i in enumerate(np.random.randint(N, size=100)):  # fix the O(N^2) here
            dX = self._x_norm(X-X[i, :])
            dY = self._y_norm(Y-Y[i, :])

            dZ = np.array([dX, dY])
            dZ = np.max(dZ, axis=0)
            dZ = np.sort(dZ)[k]

            tx = dZ - dX
            I -= digamma(np.sum(tx > 0)+1) / 100

            ty = dZ - dX
            I -= digamma(np.sum(ty > 0)+1) / 100

        return I

    def _cleanup(self):
        """
        cleanup and relabel unused clusters
        """
        # relabel clusters
        uniques = np.unique(self.f)
        unique_map = {v: i for i, v in enumerate(uniques)}
        self.f = np.vectorize(unique_map.get)(self.f)

        # remove unused bins
        self.qy_t = self.qy_t[:, np.sum(self.qy_t, axis=0) != 0]
        self.qt = self.qt[self.qt != 0]

        self.hiddens = uniques.size

# reporting and visualization #

    def report_clusters(self):
        """
        returns current cluster assignmetns
        """
        print("Found %d clusters with beta = %.2f" %
              (np.unique(self.f).size, self.beta))
        return self.f

    def visualize_clusters(self, vsz=3, debug=False):
        """
        displays clusters
        """
        qx_t = np.zeros((vsz*vsz, self.hiddens))
        finv = {x: set() for x in range(self.hiddens)}

        self.finv = finv

        for idx, element in enumerate(self.f):
            finv[element].add(idx)

        print({t: np.mean([bin(x).count('1') for x in finv[t]]) for t in finv})
        for t in finv:
            qx_t[:, t] = np.sum(
                np.array([self._bin2row(x, vsz*vsz) * self.px[x]
                          for x in finv[t]]), axis=0)

        # normalize cluster averages
        qx_t = qx_t / self.qt

        # reorder clusters by probability
        order = np.argsort(-self.qt)
        qx_t = qx_t[:, order]

        # reshape and plot
        if debug:
            clusters = np.zeros((vsz, vsz*self.hiddens))

            for t in finv:
                clusters[:, t*vsz:(t+1)*vsz] = qx_t[:, t].reshape(vsz, vsz)

            plt.matshow(clusters, cmap=plt.cm.gray)
            plt.show()

        return qx_t

    def mi_relevant(self):
        """
        returns mutual information between cluster assignments and environment
        I(T;Y)
        """
        mi = 0

        for y in range(self.ysz):
            for t in range(self.hiddens):
                mi += self.qy_t[y, t] * self.qt[t] * (
                    np.log2(self.qy_t[y, t] + DIB.eps) -
                    np.log2(self.py[y] + DIB.eps))

        return mi

    def mi_captured(self):
        """
        returns mutual information between cluster assignments and environment
        I(T;X)
        """
        mi = 0

        for x in range(self.xsz):
            for t in range(self.hiddens):
                mi += (self.f[x] == t) * self.px[x] * (
                    np.log2((self.f[x] == t) + DIB.eps) -
                    np.log2(self.qt[t] + DIB.eps))

        return mi

    def _bin2row(self, x, sz=9):
        """
        converts x to binary row (big endian)
        - this could probably be more efficient
        """
        return np.array([(x//2**k) % 2 == 1 for k in range(sz)])


# helpful functions #

def debugshow(thing):
    plt.matshow(thing, cmap=plt.cm.gray)
    plt.show()


def divide(a, b):
    """
    a / b, b==0 not used
    a better divide function
    """
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
