import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class DIB:
    eps = 1e-12

    def __init__(self, pxy, beta=5, hiddens=100):
        self.beta = beta
        self.hiddens = hiddens

        self.pxy = pxy  # joint distribution, 2D numpy array
        self.xsz, self.ysz = pxy.shape

        self.px = pxy.sum(axis=1)
        self.px = divide(self.px, self.px.sum())

        self.py_x = divide(pxy, self.px[:,np.newaxis])
        self.py_x = self.py_x.T

        self.beta = beta

        # initialize clusters
        x = np.eye(hiddens)
        self.qt_x = x[:,np.random.randint(self.hiddens, size=self.xsz)]
        self.f = np.argmax(self.qt_x, axis=0)
        self.l = np.zeros((self.xsz, hiddens))

        self.cost = 0     # effectively infinity

    def compress(self, epsilon=1e-4):
        self._update()

        prev_cost = 1e6

        idx = 0

        while abs(self.cost - prev_cost) > epsilon:
            print("Iteration %d: %.2f" % (idx, self.cost))
            prev_cost = self.cost
            self._step()
            self._update()

            idx += 1

        self._cleanup()

        return self.cost

# core DIB helper functions #
    def _step(self):
        """
        updates cluster assignments by maximizing DIB objective
        """
        self.qt_x = self.l.T     # not correct, but should work for DIB scheme

        f = np.argmax(self.qt_x, axis=0)

        # try to merge clusters
        for a, b in combinations(range(self.hiddens), 2):
            ftest = np.where(f==a, b, f)
            cost = self._calculate_cost(ftest)

            if cost > self.cost:
                f = ftest

        self.f = f

    def _update(self):
        """
        recalculates q(t) and q(t|x) for current cluster assignments
        """
        self.qt = np.zeros(self.hiddens)
        self.qy_t = np.zeros((self.ysz, self.hiddens))

        for x in range(self.xsz):
            t = self.f[x]
            self.qt[t] += self.px[x]

        for x in range(self.xsz):
            t = self.f[x]
            self.qy_t[:,t] += divide(self.pxy[x,:], self.qt[t])

        d = np.zeros((self.xsz, self.hiddens))
        for x in range(self.xsz):   # can this be simplified?
            for t in range(self.hiddens):
                for y in range(self.ysz):
                    d[x,t] += self.py_x[y,x] * (\
                            np.log(self.py_x[y,x] + DIB.eps) - \
                            np.log(self.qy_t[y,t] + DIB.eps))

        self.l = np.log(self.qt + DIB.eps) - self.beta * d

        self.cost = self._calculate_cost(self.f)

    def _calculate_cost(self, f):
        """
        calculate the cost of a proposed clustering
        """
        cost = 0
        for x in range(self.xsz):
            t = f[x]
            cost += self.l[x,t]

        return cost

    def _cleanup(self):
        """
        cleanup and relabel unused clusters
        """
        # relabel
        uniques = np.unique(self.f)
        self.f = np.vectorize({v:i for i, v in enumerate(uniques)}.get)(self.f)

        self.hiddens = uniques.size

# reporting and visualization #

    def report_clusters(self):
        """
        returns current cluster assignmetns
        """
        print("Found %d clusters with beta = %.1f" % \
                (np.unique(self.f).size, self.beta))
        return self.f

    def visualize_clusters(self, vsz=3):
        """
        displays clusters
        """
        qx_t = np.zeros((vsz*vsz, self.hiddens))  # hardcoded system size
        finv = {x: set() for x in range(self.hiddens)}

        for idx, element in enumerate(self.f):
           finv[element].add(idx)

        print({t: np.mean([bin(x).count('1') for x in finv[t]]) for t in finv})
        for t in finv:
            qx_t[:,t] = np.mean(\
                    np.array([self._bin2row(x, vsz*vsz) for x in finv[t]]),\
                    axis=0)

        clusters = np.zeros((vsz, vsz*self.hiddens))

        for t in finv:
            clusters[:,t*vsz:(t+1)*vsz] = \
                    qx_t[:,t].reshape(vsz,vsz)

        plt.matshow(clusters, cmap=plt.cm.gray)
        plt.show()

        return qx_t

    def mutual_information(self):
        """
        returns mutual information between cluster assignments and environment
        I(H;E)
        """
        pass

    def _bin2row(self, x, sz=9):
        """
        converts x to binary row (big endian)
        - this could probably be more efficient
        """
        return np.array([(x//2**k) % 2 == 1 for k in range(sz)])


# helpful functions #

def debugshow(thing):
    plt.matshow(thing,cmap=plt.cm.gray)
    plt.show()

def divide(a,b):
    """
    a / b, b==0 not used
    a better divide function
    """
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
