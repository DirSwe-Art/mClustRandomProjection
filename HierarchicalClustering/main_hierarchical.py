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
import pandas as pd
import matplotlib.pyplot as plt

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
	linkage_matrix   = pd.DataFrame({'cl1_i':[], 'cl2_i':[], 'distance':[], 'new_cl_length':[]})
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
		#print(len(clusterings), len(pool))	
		
		# identify the closest two clusters in the pool
		i1, i2, dis = closetClusters(pool, linkage, affinity)
		
		# merge them
		new_cl_id  += 1
		new_cl      = mergeTwoClusters(pool[i1], pool[i2], new_cl_id)
		
		# add a record to the linkage matrix (1st id, 2nd id, distance, merged cluster length)
		new_link    = pd.DataFrame({	'cl1_i':         [pool[i1]['cluster_i']], 
										'cl2_i':         [pool[i2]['cluster_i']], 
										'distance':      [dis], 
										'new_cl_length': [len(new_cl['elements_i'])]
									}, index=[new_cl_id])
		linkage_matrix = pd.concat( [linkage_matrix, new_link], ignore_index=False  )
		
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
	print(linkage_matrix)	
	# form a class to save the hierarchical model		
	class model:
		def __init__(self):
			self.clusterings   = clusterings
			self.linkageMatrix = linkage_matrix
			self.labels		   = labels
			self.centers	   = centers
		
	return model()


def linkColorFunction(link_id):
	n_leaves    = len(y)
	link_colors = {}
	
	def get_clusters_from_node(node_id, n_leaves):
		"""Returns a set of clusters that belong to a given node in the dendrogram."""
		if node_id < n_leaves:
			return {y[node_id]}
		left_child  = int( Z.values[node_id - n_leaves][0] )
		right_child = int( Z.values[node_id - n_leaves][1] )
		return get_clusters_from_node(left_child, n_leaves).union( get_clusters_from_node(right_child, n_leaves) )
	
	for i in range(len(Z)):
		left, right     = int(Z.values[i, 0]), int(Z.values[i, 1])
		left_clusters   = get_clusters_from_node(left, n_leaves)
		right_clusters  = get_clusters_from_node(right, n_leaves)
		merged_clusters = left_clusters.union(right_clusters)
		
		if len(merged_clusters) == 1:
			# If both children belong to the same cluster, color the link accordingly
			link_colors[i + n_leaves] = colors[next(iter(merged_clusters))]
		elif len(merged_clusters) > 1 and len(left_clusters)  == 1:
			link_colors[i + n_leaves] = colors[next(iter(left_clusters))]		
		elif len(merged_clusters) > 1 and len(right_clusters) == 1:
			link_colors[i + n_leaves] = colors[next(iter(right_clusters))]
		else:
			# Otherwise, color it grey to indicate it merges different clusters
			link_colors[i + n_leaves] = 'grey'

	return link_colors.get(link_id, 'grey')
	
def plotDendrogram(Z, **kwargs):
	plt.close('all')
	plt.figure(figsize=(10,6))
	
	Z2           = np.array(copy.deepcopy(Z))
	Z2[:, 2]     = [ float(i)/max(Z2[:, 2]) for i in Z2[:, 2] ]

	denZ = dendrogram( Z2,
				   #color_threshold       = 0,
				   #above_threshold_color = 'grey',
				   #link_color_func       = linkColorFunction,
				   leaf_rotation         = 90,
				   leaf_font_size        = 10,
				   **kwargs
				   )
	
	ax = plt.gca()
	x_labels = ax.get_xmajorticklabels()
	for lbl, leaf_idx in zip(x_labels, denZ['leaves']):
		lbl.set_color(leaf_colors[leaf_idx])
	
	'''	
	# when labels are not used
	ax = plt.gca()
	x_labels = ax.get_xmajorticklabels()
	for lbl in x_labels:
		lbl.set_color(leaf_colors[int(lbl.get_text())])
	'''
	
	plt.title('Dendrogram with Cluster-based Link and Leaf Colors')
	plt.xlabel('Sample index')
	plt.ylabel('Normalized Distance')
	plt.show()
	

	
	

# ==================================================================
import random

X = []
M = [[3,3],[9,9]]

for m in M:
    X += np.random.multivariate_normal(m, np.identity(2)/2, size=20).tolist()

random.shuffle(X)
X = np.array(X)
X = X - np.mean(X)

plt.figure(figsize=(10,6))
plt.scatter( *zip(*X) )
plt.show()



linkage='average'
affinity='euclidean'

mdl = hierarchical(X, n_clusters=2, linkage=linkage, affinity=affinity)


while True:
	try:
		clust_k   = str(input('    Enter the number of clusters you want to view:'))
		clust_    = [ clust for clust in mdl.clusterings if clust['clusters_k'] == int(clust_k)]
		clust_x   = clust_[0]['clusters_x']
		clust_i   = clust_[0]['clusters_i']
		Z         = mdl.linkageMatrix
		
		y, C      = outputLC(X, clust_i)	

		colors      = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
		
		import matplotlib.colors as mclr
		plt.close('all')
		plt.figure(figsize=(10,6))
		for i, cl in enumerate(clust_x):
			F1, F2 = zip(*cl)
			plt.scatter( F1, F2, [ mclr.to_rgb(colors[i]) for i in y ] )
		plt.show()
		
		leaf_colors = {i: colors[lebel] for i, lebel in enumerate(y)}
		

		
		

		
		

	
		plotDendrogram(Z, labels=y)
		
		'''	
		C = centers(clust_x)
		clusteredImage = np.array([ C[i] for i in Y ])
		display_Image(DATA, clusteredImage, imRow, imCol, imDim)
		'''

	except ValueError:
		if clust_k == 'q': print("\nProgram is ended"); break
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



	
# ==================================================================
