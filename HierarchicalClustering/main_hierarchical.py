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
	
	def euc(a,b): 
		return math.sqrt( sum( [ [float(a[i]) - float(b[i])]** for i in range(len(a))] ) )
	
	def man(a,b): 
		return sum(       abs( [ [a[i] - b[i]]** for i in range(len(a))] ) ) 
	
	def dist(a,b):
		if affinity == 'euclidean': return euc(a,b)
		if affinity == 'manhatten': return man(a,b)
	
	def initialClusters(DATA):
		return [ {'cluster_ID':   i},
				 {'elements_IDs':[i]},
				 {'elements':    [x]},
				 {'used':        '1'} for i, x in enumerate(DATA) ]

	def findClosetClusters(clustersDict, method, metric):
		S         = []
		
		comb_ids  = list( combinations(range(len(clustersDict)),2) )
		if method == 'centroid':
			for i1, i2 in comb_ids:
				mu1 = [ np.mean(col) for col in zip(*clustersDict[i1]['elements']) ]
				mu2 = [ np.mean(col) for col in zip(*clustersDict[i2]['elements']) ]
				S.append([i1, i2, dist(mu1, mu2)])
			minDistID = np.argmin(np.array(S)[:,2])
			return S[minDistID][0], S[minDistID][1], S[minDistID][2]]
		else:
			for i1, i2 in comb_ids:
				m   = len(clustersDict[i1]['elements'])
				n   = len(clustersDict[i2]['elements'])
				
				if m == 1 and n == 1:
					S.append([i1, i2, dist(clustersDict[i1]['elements'][0], clustersDict[i2]['elements'][0])])
				else:
					S2= np.zeros((m,n))
					for i in range(m):
						for j in range(n):
							S2[i][j] = dist( clustersDict[i1]['elements'][i], clustersDict[i1]['elements'][j] )
						if method   == 'single'  : S.append([i1, i2, np.amin(S2)])
						elif method == 'complete': S.append([i1, i2, np.amax(S2)])
						elif method == 'average' : S.append([i1, i2, np.mean(S2)])
