from anndata import AnnData
from collections import Counter
import datetime
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scanpy as sc
import scipy.stats as ss
import sys
import warnings

np.random.seed(1)
random.seed(1)

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
