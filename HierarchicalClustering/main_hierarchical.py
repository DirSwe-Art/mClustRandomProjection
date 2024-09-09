"""Hierarchical Clustering From Scratch Implementation"""

# Authors: DirarSweidan
# License: DSB 3-Claus

'''
This function returns a hierarchical clustering class model that containes model.clutserings, model.linkagematrix, model.labels, and model.centers 
'''

from itertools import combinations
import numpy as np
import math
import copyfrom scipy.cluster.hierarchy import dendrogram

#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from scipy.linalg import fractional_matrix_power


# ========================================================================

def hierarchical(DATA, n_clusters=2, linkage='average', affinity='euclidean'):
	def euc(a,b): return math.sqrt( sum( [ [a[i] - b[i]]** for i in range(len(a))] ) )
	def man(a,b): return sum(       abs( [ [a[i] - b[i]]** for i in range(len(a))] ) ) 
	

