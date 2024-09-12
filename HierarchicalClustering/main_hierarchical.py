"""Hierarchical Clustering From Scratch Implementation"""

# Authors: DirarSweidan
# License: DSB 3-Claus

'''
This function returns a hierarchical clustering class model that containes model.clutserings, model.linkagematrix, model.labels, and model.centers 
'''

import numpy as np
import math, copy
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram

#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from scipy.linalg import fractional_matrix_power


# ========================================================================

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

def closetClusters(clusters, method, affinity):
	S           = []
	comb_ids    = list( combinations(range(len(clusters)),2) )
	
	if method   == 'centroid':
		for i1, i2 in comb_ids:
			mu1 = [ np.mean(col) for col in zip(*clusters[i1]['elements_x']) ]
			mu2 = [ np.mean(col) for col in zip(*clusters[i2]['elements_x']) ]
			S.append([i1, i2, dist(mu1, mu2)])
		
		min_id= np.argmin(np.array(S)[:,2])
		return S[min_id][0], S[min_id][1], S[min_id][2]
	else:
		for i1, i2 in comb_ids:
			m   = len(clusters[i1]['elements_x'])
			n   = len(clusters[i2]['elements_x'])
			if m == 1 and n == 1:
				cl1 = clusters[i1]['elements_x'][0]
				cl2 = clusters[i2]['elements_x'][0]
				S.append([i1, i2, dist(cl1, cl2)])
			else:
				S2= np.zeros((m,n))
				for i in range(m):
					for j in range(n):
						cl1 = clusters[i1]['elements_x'][i]
						cl2 = clusters[i2]['elements_x'][j]
						S2[i][j] = dist(cl1, cl2)
				if method == 'single'  : S.append([i1, i2, np.amin(S2)])
				elif method == 'complete': S.append([i1, i2, np.amax(S2)])
				elif method == 'average' : S.append([i1, i2, np.mean(S2)])
		min_id = np.argmin(np.array(S)[:,2])
		return S[min_id][0], S[min_id][1], S[min_id][2]

def mergeTwoClusters(cl1, cl2, new_cl_id):
	new_cl = {'cluster_i':  int(new_cl_id),
			  'elements_x': cl1['elements_x'] + cl2['elements_x'],
			  'elements_i': cl1['elements_i'] + cl2['elements_i']
			  }
	return new_cl

def outputLabels(X, clusters_i):
	labels  = np.zeros(len(X))
	for k, cl_ids in enumerate(clusters_i):
		labels[cl_ids] = k
	return labels

def outputCenters(X, clusters_i):
	centers = []
	for cl_ids in clusters_i:
		centers.append([ np.mean(col) for col in zip(*X[cl_ids]) ])
	return centers

def outputLC(X, clusters_i):
	labels  = np.zeros(len(X), dtype=int)
	centers = [ [] for i in range(len(clusters_i)) ]
	for k, cl_ids in enumerate(clusters_i):
		labels[cl_ids] = k
		centers[k]     = [ np.mean(col) for col in zip(*X[cl_ids]) ]
	return np.array(labels), centers

def hierarchical(DATA, n_clusters=2, linkage='average', affinity='euclidean'):
	X                = copy.deepcopy(DATA)
	pool             = initialClusters(X)
	clusterings      = []
	linkage_matrix   = []
	labels           = []
	centers          = []
	
	clusterings.append( {'clusters_l': 0,
						 'clusters_k': len(pool),
						 'clusters_x': [cl['elements_x'] for cl in pool],
						 'clusters_i': [cl['elements_i'] for cl in pool]
							  } )
						  
	new_cl_id	 	= len(pool)-1 # the last index in the pool
	new_clust_id	= 0			  # the last index in the clusterings
	while len(pool) > 1:
		print(len(clusterings), len(pool))	
		
		# identify the closest two clusters in the pool
		i1, i2, dis = closetClusters(pool, linkage, affinity)
		
		# merge them
		new_cl_id  += 1
		new_cl      = mergeTwoClusters(pool[i1], pool[i2], new_cl_id)
		
		# add a record to the linkage matrix (1st id, 2nd id, distance, merged cluster length)
		linkage_matrix.append([pool[i1]['cluster_i'], pool[i2]['cluster_i'], dis, len(new_cl['elements_i']) ])#, new_cl_id])
		
		# remove the identified closest two clusters from the pool
		for id in sorted([i1, i2], reverse=True): 
			del pool[id]
		
		# append the new merged cluster to the pool 
		pool.append(new_cl)
		
		# form a new clustering dict containes the pool clusters and add it to the clusterings list
		new_clust_id+= 1
		new_clust	 = {'clusters_l': new_clust_id,
						'clusters_k': len(pool),
						'clusters_x': [cl['elements_x'] for cl in pool],
						'clusters_i': [cl['elements_i'] for cl in pool]
						}
		clusterings.append(new_clust)
		
		# output the required clustering's labels and centers
		if new_clust['clusters_k'] == n_clusters: 
			labels, centers = outputLC(X, new_clust['clusters_i'])	
		
	# form a class to save the hierarchical model		
	class model:
		def __init__(self):
			self.clusterings   = clusterings
			self.linkageMatrix = linkage_matrix
			self.labels		   = labels
			self.centers	   = centers
		
	return model()

def plotDendrogram(Z, **kwargs):
	linkage_mat = np.array(copy.deepcopy(Z))
	linkage_mat[:, 2] = [ float(i)/max(linkage_mat[:, 2]) for i in linkage_mat[:, 2] ]
	plt.figure(figsize=(10,6))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('Sample index')
	plt.ylabel('Normalized Distance')
	dendrogram( linkage_mat,
				leaf_rotation = 90,
				leaf_font_size = 8,
				#truncate_mode='lastp',
				#show_contracted=True,
				**kwargs
				)
	plt.show()


	
# ==================================================================
import random
import matplotlib.pyplot as plt

X = []
M = [[3,3],[9,9]]

for m in M:
    X += np.random.multivariate_normal(m, np.identity(2)/2, size=20).tolist()

random.shuffle(X)
X = np.array(X)
X = X - np.mean(X)
plt.scatter( *zip(*X) )
plt.show()



linkage='average'
affinity='euclidean'

mdl = hierarchical(X, n_clusters=2, linkage=linkage, affinity=affinity)


while True:
	try:
		clust_k   = str(input('    Enter the number of clusters you want to view:'))
		if clust_k == 'q': break
		clust_    = [ clust for clust in mdl.clusterings if clust['clusters_k'] == int(clust_k)]
		clust_x   = clust_[0]['clusters_x']
		clust_i   = clust_[0]['clusters_i']
		
		y, C      = outputLC(X, clust_i)	
		
		for i, cl in enumerate(clust_x):
			F1, F2 = zip(*cl)
			plt.scatter( F1, F2)
		
		plotDendrogram(mdl.linkageMatrix, labels=y)
		
		'''	
		C = centers(clust_x)
		clusteredImage = np.array([ C[i] for i in Y ])
		display_Image(DATA, clusteredImage, imRow, imCol, imDim)
		'''

	except ValueError:
		if numberOfClusters == 'q': print("\nProgram is ended"); break
		print("Invalid number of clusters")
'''
clr = ['g','b']
plt.scatter( *zip(*X), color= [clr[i] for i in mdl.labels] )
plt.show()
plt.close()

Z = mdl.linkageMatrix
print(Z)

#plt.figure()
#dn = dendrogram(Z)
plotDendrogram(Z, labels= mdl.labels)
'''