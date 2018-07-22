import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

from information_bottleneck import divide

# parameters
show_plot = True

# load data
ip_files = sorted(
    glob.glob("/Users/andrew/Documents/rgml/ip_data/vanilla2/ipdata_*.pkl"))
bsizes = [int(x[-6:-4]) for x in ip_files]

ipdata = {}

for idx, ip_file in enumerate(ip_files):
    with open(ip_file, 'rb') as f:
        # I(X;T), I(Y;T), beta, clusters, angles
        ipdata[bsizes[idx]] = pickle.load(f)
        print("Loaded %s" % ip_file)

# plot
if show_plot:
    # DIB plot
    plt.figure()
    for bsize in bsizes:
        plt.plot(ipdata[bsize]['info_x'], ipdata[bsize]['info_y'], 'o')
    plt.title('DIB Plane Plot')
    plt.xlabel('H(T)')
    plt.ylabel('I(Y;T)')
    plt.legend(["%dx%d block" % (x, x) for x in bsizes])

    plt.show()

    # clusters as cumulative relevant information plot
    num_clusters = 10
    max_infos = np.zeros((len(bsizes), num_clusters))

    # get mapping from cluster number to max info_y
    for bidx, bsize in enumerate(bsizes):
        for idx, nc in enumerate(ipdata[bsize]['clusters2']):
            if nc > num_clusters:
                break
            if max_infos[bidx, nc-1] < ipdata[bsize]['info_y'][idx]:
                max_infos[bidx, nc-1] = ipdata[bsize]['info_y'][idx]
        max_infos[bidx, :] /= max(ipdata[bsize]['info_y'])

    # fill in cluster gaps
    for bidx in range(len(bsizes)):
        for cidx in range(1, num_clusters):
            if max_infos[bidx, cidx] < max_infos[bidx, cidx-1]:
                max_infos[bidx, cidx] = max_infos[bidx, cidx-1]

    marg_infos = np.diff(max_infos)

    # create figure and plot
    clusters = np.arange(2, num_clusters+1)

    f, axarr = plt.subplots(len(bsizes), sharex=True, sharey=True)

    for bidx, bsize in enumerate(bsizes):
        axarr[bidx].bar(clusters, marg_infos[bidx,:])
        axarr[bidx].set_ylabel("%dx%d" % (bsize, bsize))



    axarr[0].set_title('Normalized marginal information')
    axarr[-1].set_xlabel('number of clusters')
    plt.xticks(clusters)
    plt.show()

    # kink angle plot
    # clusters = np.arange(1, 10)

    # f, axarr = plt.subplots(len(bsizes), sharex=True, sharey=True)

    # for idx, bsize in enumerate(bsizes):
    #     axarr[idx].bar(clusters, ipdata[bsize]['theta'][:len(clusters)])
    #     axarr[idx].set_ylabel("%dx%d" % (bsize, bsize))

    # axarr[0].set_title('Kink Angles')
    # axarr[-1].set_xlabel('number of clusters')
    # plt.xticks(clusters)
    # plt.show()
