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
import numpy as np
import math
from itertools import combinations

def euc(a,b): 
	return math.sqrt(sum([ (float(a[i]) - float(b[i]))**2 for i in range(len(a)) ])) 

def man(a,b):
	return sum([ abs(float(a[i]) - float(b[i])) for i in range(len(a))])

def dist(a,b):
	if affinity == 'euclidean': return euc(a,b)
	if affinity == 'manhatten': return man(a,b)

def initialClusters(DATA):
	return [ {'cluster_i':   i,
			  'elements_x': [x],
			  'elements_i': [i] } for i, x in enumerate(DATA) ]

def closetClusters(clustersDict, method, affinity):
	S           = []
	comb_ids    = list( combinations(range(len(clustersDict)),2) )
	
	if method   == 'centroid':
		for i1, i2 in comb_ids:
			mu1 = [ np.mean(col) for col in zip(*clustersDict[i1]['elements_x']) ]
			mu2 = [ np.mean(col) for col in zip(*clustersDict[i2]['elements_x']) ]
			S.append([i1, i2, dist(mu1, mu2)])
		
		mDistID = np.argmin(np.array(S)[:,2])
		return S[mDistID][0], S[mDistID][1], S[mDistID][2]
	else:
		for i1, i2 in comb_ids:
			m   = len(clustersDict[i1]['elements_x'])
			n   = len(clustersDict[i2]['elements_x'])
			if m == 1 and n == 1:
				clust1 = clustersDict[i1]['elements_x'][0]
				clust2 = clustersDict[i2]['elements_x'][0]
				S.append([i1, i2, dist(clust1, clust2)])
			else:
				S2= np.zeros((m,n))
				for i in range(m):
					for j in range(n):
						S2[i][j] = dist( clustersDict[i1]['elements_x'][i], clustersDict[i2]['elements_x'][j] )
				if method   == 'single'  : S.append([i1, i2, np.amin(S2)])
				elif method == 'complete': S.append([i1, i2, np.amax(S2)])
				elif method == 'average' : S.append([i1, i2, np.mean(S2)])
		mDistID = np.argmin(np.array(S)[:,2])
		return S[mDistID][0], S[mDistID][1], S[mDistID][2]

def outputLC(X, clusters_i):
	labels  = np.zeros(len(X))
	centers = []
	for k, cluster in enumerate(clusters_i):
		labels[clusters_i] = k
		centers[k]         = [ np.mean(col) for col in zip(*X[cluster]) ]
	return labels, labels
	
def computeCenters(clusters_x):
	
	for k, cluster in enumerate(clusters_x):
		
	return centers

def hierarchical(DATA, n_clusters=2, linkage='average', affinity='euclidean'):
	X                = copy.deepcopy(DATA)
	clusters_pool    = initialClusters(X)
	clusterings_pool = []
	linkage_matrix   = []
	
	clusterings_pool.append( {'clusters_l':    0,
							  'clusters_k': len(clusters_pool),
							  'clusters_x': [cl['elements_x'] for cl in clusters_pool],
							  'clusters_i': [cl['elements_i'] for cl in clusters_pool]
							  } )
							  
	new_cl_id	 	= len(clusters_pool)
	new_clust_id	= 0
	while len(clusters_pool) > 1:
		i1, i2, dis = closetClusters(clusters_pool, linkage, affinity)
		
		new_cl_id  += 1
		new_cl      = {'cluster_i':  new_cl_id,
					   'elements_x': clusters_pool[i1]['elements_x'] + clusters_pool[i2]['elements_x'],
					   'elements_i': clusters_pool[i1]['elements_i'] + clusters_pool[i2]['elements_i']
					   }
		clusters_pool.pop(il)
		clusters_pool.pop(i2)
		clusters_pool.append(new_cl)
		
		new_clust_id+= 1
		new_clust	 = {'clusters_l': new_clust_id,
						'clusters_k': len(clusters_pool),
						'clusters_x': [cl['elements_x'] for cl in clusters_pool],
						'clusters_i': [cl['elements_i'] for cl in clusters_pool]
						}
		if new_clust['clusters_k'] == n_clusters: 
			labels, centers = outputLC(X, new_clust['clusters_x'])
		
		
		linkage_matrix.append([i1, i2, dis, new_cl['elements_x'])
		clusterings_pool.append()
		
		class model:
			def __init__(self):
				self.clusterings   = clusterings_pool
				self.linkageMatrix = np.array(linkage_matrix)
				self.labels		   = labels
				self.centers	   = centers
		
		return model()