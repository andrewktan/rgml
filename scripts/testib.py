import matplotlib.pyplot as plt
import numpy as np
from information_bottleneck import *

# learning 3 input xor
pxy = np.array([[1, 0],  # 000
                [0, 1],  # 001
                [0, 1],  # 010
                [1, 0],  # 011
                [0, 1],  # 100
                [1, 0],  # 101
                [1, 0],  # 110
                [0, 1]]  # 111
               )

pxy = np.abs(pxy + np.random.rand(*pxy.shape) * 0.1)

pxy = pxy / np.sum(pxy)

dib = DIB(pxy, beta=10, hiddens=100)
dib.compress()

dib.report_clusters()
