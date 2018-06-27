import matplotlib.pyplot as plt
import numpy as np

from dataset_iterator import *
from layered_coarse_grain import *


class IsingDataset(DatasetIterator):
    """
    iterator for Ising
    """
