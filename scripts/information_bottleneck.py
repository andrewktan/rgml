import numpy as np
import matplotlib.pyplot as plt

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


    def compress(self, epsilon=1e-4):
        # init
        self._update()

        cost_prev = 1e6
        cost = 1e5

        idx = 0

        while (cost_prev - cost) > epsilon:
            print("Iteration %d: %.2f" % (idx, cost))
            cost_prev = cost
            cost = self._step()
            self._update()

            idx += 1

        self._cleanup()

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

    def _step(self):
        """
        updates cluster assignments by maximizing DIB objective
        """
        d = np.zeros((self.xsz, self.hiddens))
        for x in range(self.xsz):   # can this be simplified?
            for t in range(self.hiddens):
                for y in range(self.ysz):
                    d[x,t] += self.py_x[y,x] * (np.log(self.py_x[y,x] + DIB.eps) - np.log(self.qy_t[y,t] + DIB.eps))

        l = np.log(self.qt + DIB.eps) - self.beta * d
        #l = -d

        self.qt_x = l.T     # not correct, but should work for current DIB scheme

        self.f = np.argmax(self.qt_x, axis=0)

        self.d = d
        self.l = l

        cost = 0
        for x in range(self.xsz):
            t = self.f[x]
            cost += l[x,t]

        return cost
   
    def _cleanup(self):
        """
        cleanup and relabel unused clusters
        """
        # relabel
        uniques = np.unique(self.f)
        self.f = np.vectorize({v:i for i, v in enumerate(uniques)}.get)(self.f)

        self.hiddens = uniques.size

    def report(self):
        """
        returns current cluster assignmetns
        """
        print("Found %d clusters with beta = %.1f" % (np.unique(self.f).size, self.beta))
        return self.f

    def visualize_clusters(self):
        """
        displays clusters
        """
        finv = {x: set() for x in range(self.hiddens)}

        for idx, element in enumerate(self.f):
            finv[element].add(idx)

        avg_counts = {c: np.mean([bin(x).count('1') for x in finv[c]]) for c in range(len(finv))}

        print(avg_counts)


def debugshow(thing):
    plt.matshow(thing,cmap=plt.cm.gray)
    plt.show()

def divide(a,b):
    """
    a / b, b==0 not used
    a better divide function
    """
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
