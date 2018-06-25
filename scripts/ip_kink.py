import pickle

import matplotlib.pyplot as plt
import numpy as np

from information_bottleneck import divide

# parameters
show_plot = True

# load data
with open('ip_data.npy', 'rb') as f:
    x5, y5, x7, y7, x9, y9 = pickle.load(f)

# plot
if show_plot:
    plt.plot(x5, y5, 'ko', x7, y7, 'ro', x9, y9, 'go')
    plt.title('Information Plane Plot')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')
    plt.legend(['5x5 block', '7x7 block', '9x9 block'])

plt.show()

# calculate angles
x5 = np.round(x5, decimals=2)
x7 = np.round(x7, decimals=2)
x9 = np.round(x9, decimals=2)
y5 = np.round(y5, decimals=2)
y7 = np.round(y7, decimals=2)
y9 = np.round(y9, decimals=2)

dydx5 = divide(np.diff(y5), np.diff(x5))
dydx7 = divide(np.diff(y7), np.diff(x7))
dydx9 = divide(np.diff(y9), np.diff(x9))

dydx5 = dydx5[dydx5 != 0]
dydx7 = dydx7[dydx7 != 0]
dydx9 = dydx9[dydx9 != 0]

dslope5 = dydx5[1] - dydx5[0]
dslope7 = dydx7[1] - dydx7[0]
delope9 = dydx9[1] - dydx9[0]
