import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

from information_bottleneck import divide

# parameters
show_plot = True

# load data
ip_files = sorted(glob.glob("ipdata_*.pkl"))
bsizes = [int(x[7:9]) for x in ip_files]

ip_data = {}

for idx, ip_file in enumerate(ip_files):
    with open(ip_file, 'rb') as f:
        # I(X;T), I(Y;T), beta, clusters, angles
        ip_data[bsizes[idx]] = pickle.load(f)
        print("Loaded %s" % ip_file)

# plot
if show_plot:
    plt.figure()
    for bsize in bsizes:
        plt.plot(ip_data[bsize]['info_x'], ip_data[bsize]['info_y'], 'o')
    plt.title('DIB Plane Plot')
    plt.xlabel('H(T)')
    plt.ylabel('I(Y;T)')
    plt.legend(["%dx%d block" % (x, x) for x in bsizes])

    plt.show()

    # calculate angles
    clusters = np.arange(1, 10)

    f, axarr = plt.subplots(len(bsizes), sharex=True, sharey=True)

    for idx, bsize in enumerate(bsizes):
        axarr[idx].bar(clusters, ip_data[bsize]['theta'][:len(clusters)])
        axarr[idx].set_ylabel("%dx%d" % (bsize, bsize))

    axarr[0].set_title('Kink Angles')
    axarr[-1].set_xlabel('number of clusters')
    plt.xticks(clusters)
    plt.show()
