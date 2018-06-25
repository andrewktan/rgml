import pickle

import matplotlib.pyplot as plt
import numpy as np

from information_bottleneck import divide

# parameters
show_plot = True

# load data
with open('ip_data.npy', 'rb') as f:
    # I(X;T), I(Y;T), beta, clusters, angles
    x5, y5, b5, c5, a5, x7, y7, b7, c7, a7, x9, y9, b9, c9, a9 = pickle.load(f)

# plot
if show_plot:
    plt.plot(x5, y5, 'ko', x7, y7, 'ro', x9, y9, 'go')
    plt.title('Information Plane Plot')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.legend(['5x5 block', '7x7 block', '9x9 block'])

plt.show()

# calculate angles
clusters = np.arange(1, 10)
theta5 = np.array([a5[k] for k in clusters])
theta7 = np.array([a7[k] for k in clusters])
theta9 = np.array([a9[k] for k in clusters])

plt.bar(clusters, theta5)

f, axarr = plt.subplots(3, sharex=True, sharey=True)
axarr[0].bar(clusters, theta5)
axarr[1].bar(clusters, theta7)
axarr[2].bar(clusters, theta9)

axarr[0].set_title('Kink Angles')
axarr[0].set_ylabel('theta 5x5 [deg]')
axarr[1].set_ylabel('theta 7x7 [deg]')
axarr[2].set_ylabel('theta 9x9 [deg]')
axarr[2].set_xlabel('number of clusters')
plt.xticks(clusters)
plt.show()
