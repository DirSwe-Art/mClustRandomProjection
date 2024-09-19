"""Multiple Clustering via Random Projection"""

# Authors: DirarSweidan
# License: DSB 3-Claus

from sklearn import random_projection
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random, copy, scipy

# ========================================================================

def dist_clusterings(Ya, Yb):
	d          = 0
	
	comb_ids   = list( combinations(range(len(Ya), 2)) )
	for i1, i2 in comb_ids:
		if (Ya[i1]==Ya[i2] and Yb[i1]!=Yb[i2]) or (Yb[i1]==Yb[i2] and Ya[i1]!=Ya[i2]):
			d += 1
	
	return d
	
def approximate_dist_clusterings(Ya, Yb, th=100):
	if th > len(Ya): return dist_clusterings(Ya, Yb)
	
	ds_rand_Ys    = []
	for i in range(5):
		rand_ids = np.random.choice(range(len(Ya)), th, replace=False) # replace=False a value a is selected once.
		ds_rand_Ys.append( dist_clusterings([Ya[id] for id in rand_ids], [Yb[id] for id in rand_ids]) )
	
	return np.mean(ds_rand_Ys)

def affinity(data, metric=metric):
	return pairwise_distances(data, metric=metric)
	
def central(clusterings):
	# returns a clustering that has the minimum sum of distnaces with all other clutserings in the pool. #
	
	A      = affinity(clusterings)
	id_min = np.argmin([ sum(A[row]) for row in A ])
	
	return clusterings[id_min]

def ensembeled(clusterings):
	# returns a clustering where the label of each data point is the majority voting of its labels among all clusterings. #
	
	zipped_clusterings = list(zip(*clusterings))
	
	labels_majority    = []
	for col in zipped_clusterings:
		lebel, count   = Counter(col).most_common()[0]
		labels_majority.append(label)
	
	return labels_majority

def aggregated(clusterings, k):
	# returns a clustering where the label of each data point is estimated from NxN matrix of pairwisw number of clusterings two points occured in the same cluster. #
	
	ids = list(range( len(clusterings[0]) ))
	nS  = np.zeros( (len(ids), len(ids)) )
	
	for i in ids:
		for j in ids:
			count  = len([ 1 for Y in clusterings if Y[i]==Y[j] ])
			nS[i,j] = count
	
	return GaussianMixture(n_components = k).fit_predict(nS).tolsit()

def randProjClusterings(X, n_clusters, n_clusterings, n_projections, representation_method='aggregated' ):
	P = []
	for i in range(n_projections):
		projX = random_projection.GaussianRandomProjection(n_components=X.shape[1]).fit_transform(X)
		clstX = GaussianMixture(n_components=n_clusters).fit(projX)
		S     = clstX.perdict(projX)