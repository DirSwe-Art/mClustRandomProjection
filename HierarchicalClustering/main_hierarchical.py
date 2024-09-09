"""Hierarchical Clustering From Scratch Implementation"""

# Authors: DirarSweidan
# License: DSB 3-Claus

'''
This function returns a hierarchical clustering class model that containes model.clutserings, model.linkagematrix, model.labels, and model.centers 
'''

import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power


# ========================================================================

def hierarchical(DATA, n_clusters=2, linkage='average', affinity='euclidean'):
	

