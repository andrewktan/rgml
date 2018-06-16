import matplotlib.pyplot as plt
import numpy as np
from layered_coarse_grain import *

# parameters #
##############

perform_demo = True
perform_beta_sweep = False

dspath = '/Users/andrew/Documents/rgml/ising_data/'
dsname = 'data_0_45'
savefile = 'ising_ib_joint_81.npy'
sz = 81     # size of the samples (sq)
vsize = 3   # size of visible block (sq)
stride = 3
bsize = 4   # size of buffer (sq)
esize = 4   # environment size
tsize = 1000000   # table size

# information bottleneck test #
###############################

if perform_demo:
    model = LayeredCoarseGrain(dsname, dspath, 1, beta=7, debug=True)
    model.run()

    # model.get_ib_object(0).visualize_clusters(debug=True)

# beta sweep #
##############

if perform_beta_sweep:
    betas = np.linspace(0, 15, 31)
    info_y = np.zeros_like(betas)
    info_x = np.zeros_like(betas)

    for i, beta in enumerate(betas):
        dib = DIB(thist, beta=beta, hiddens=100)
        dib.compress()
        dib.report_clusters()
        info_y[i] = dib.mi_relevant()
        info_x[i] = dib.mi_captured()

    plt.plot(info_x, info_y, 'k-')
    plt.title('Information Plane Plot')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.show()
