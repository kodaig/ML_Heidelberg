from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
import numpy
from . toy_dataset import *

def norm01(data):
    res = data.astype('float32')
    res -= res.min()
    res /= res.max()
    return res