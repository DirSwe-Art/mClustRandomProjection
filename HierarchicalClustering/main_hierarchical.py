"""Hierarchical Clustering From Scratch Implementation"""

# Authors: DirarSweidan
# License: DSB 3-Claus

'''
This function returns a hierarchical clustering class model that containes model.clutserings, model.linkagematrix, model.labels, and model.centers 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, copy, random, os
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram

# ========================================================================

def euc(a,b): 
	return math.sqrt(sum([ (float(a[i]) - float(b[i]))**2 for i in range(len(a)) ])) 

def man(a,b):
	return sum([ abs(float(a[i]) - float(b[i])) for i in range(len(a))])

def dist(a,b, metric='euclidean'):
	if metric == 'euclidean': return euc(a,b)
	if metric == 'manhatten': return man(a,b)

def initialClusters(DATA):
	return [ {'cluster_i':   i,
			  'elements_x': [x],
			  'elements_i': [i] } for i, x in enumerate(DATA) ]

def closetClusters(clusters, method, metric):
	S           = []
	comb_ids    = list( combinations(range(len(clusters)),2) )
	
	if method   == 'centroid':
		for i1, i2 in comb_ids:
			mu1 = [ np.mean(col) for col in zip(*clusters[i1]['elements_x']) ]
			mu2 = [ np.mean(col) for col in zip(*clusters[i2]['elements_x']) ]
			S.append([i1, i2, dist(mu1, mu2, metric)])
		
		min_id= np.argmin(np.array(S)[:,2])
		return S[min_id][0], S[min_id][1], S[min_id][2]
	else:
		for i1, i2 in comb_ids:
			m   = len(clusters[i1]['elements_x'])
			n   = len(clusters[i2]['elements_x'])
			if m == 1 and n == 1:
				cl1 = clusters[i1]['elements_x'][0]
				cl2 = clusters[i2]['elements_x'][0]
				S.append([i1, i2, dist(cl1, cl2, metric)])
			else:
				S2= np.zeros((m,n))
				for i in range(m):
					for j in range(n):
						cl1 = clusters[i1]['elements_x'][i]
						cl2 = clusters[i2]['elements_x'][j]
						S2[i][j] = dist(cl1, cl2, metric)
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

def hierarchical(DATA, n_clusters=2, linkage='average', metric='euclidean'):
	X                = copy.deepcopy(DATA)
	pool             = initialClusters(X)
	clusterings      = []
	linkage_matrix   = pd.DataFrame({'cl1_i':[], 'cl2_i':[], 'distance':[], 'size':[]})
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
		#os.system('cls' if os.name == 'nt' else "printf '\033c'") # to clear what printed from the screen
		print(len(clusterings), len(pool), end=' ', flush=True)	
		
		# identify the closest two clusters in the pool
		i1, i2, dis = closetClusters(pool, linkage, metric)
		
		# merge them
		new_cl_id  += 1
		new_cl      = mergeTwoClusters(pool[i1], pool[i2], new_cl_id)
		
		# add a record to the linkage matrix (1st id, 2nd id, distance, merged cluster length)
		new_link    = pd.DataFrame({	'cl1_i':         [pool[i1]['cluster_i']], 
										'cl2_i':         [pool[i2]['cluster_i']], 
										'distance':      [dis], 
										'size':   [len(new_cl['elements_i'])]
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
	#print(linkage_matrix)	
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
			return {y[int(node_id)]}
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
		#elif len(merged_clusters) > 1 and len(left_clusters)  == 1:
		#	link_colors[i + n_leaves] = colors[next(iter(left_clusters))]		
		#elif len(merged_clusters) > 1 and len(right_clusters) == 1:
		#	link_colors[i + n_leaves] = colors[next(iter(right_clusters))]
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
				   leaf_rotation         = 0,
				   leaf_font_size        = 10,
				   **kwargs,
				   # customized link color based on give labels
				   color_threshold       = 0,
				   above_threshold_color = 'grey',
				   link_color_func       = linkColorFunction,

				   )
	# customized label color based on the given labels
	ax = plt.gca()
	x_labels = ax.get_xmajorticklabels()
	for lbl, leaf_idx in zip(x_labels, denZ['leaves']):
		lbl.set_color(leaf_colors[leaf_idx])
	
	'''	
	# labels and customized colors are not used
	ax = plt.gca()
	x_labels = ax.get_xmajorticklabels()
	for i, lbl in enumerate(x_labels):
		lbl.set_color(denZ['leaves_color_list'][i])
	'''
	
	plt.title('Dendrogram with Cluster-based Link and Leaf Colors')
	plt.xlabel('Sample index')
	plt.ylabel('Normalized Distance')
	plt.savefig(r'results/dendrogram_'+str(k)+'.jpg')
	plt.show()
	
# ========================================================================
def randomData():
	import os
	if not os.path.exists('results'): os.makedirs('results')
	
	X = []
	M = [[2, 2],[-2, 2],[-2, -2],[2, -2]]
	for m in M:
		X += np.random.multivariate_normal(m, np.identity(2)/3, size=20).tolist()
	
	random.shuffle(X)
	X = np.array(X)
	X = X - X.mean(axis=0)					# center the data
	
	plt.scatter( *zip(*X) )
	plt.show()	
	plt.close('all')	
	
	return X
	
def imageData(file):
	import os
	if not os.path.exists('results'): os.makedirs('results')
	IMG = plt.imread(file)					# uint8 data type
	imRow, imCol, imDim = IMG.shape
	X = []
	
	for r in range(imRow):
		for c in range(imCol):
			X.append( IMG[r][c] )
	
	X = np.array(X, dtype='uint8')	
	return X, imRow, imCol, imDim

def imageDisplay(X, XX, imR, imC, imD, text=''):
	IMG_X  = np.array(copy.deepcopy(X), dtype='uint8').reshape(imR, imC, imD)
	IMG_XX = np.array(copy.deepcopy(XX), dtype='uint8').reshape(imR, imC, imD)
	
	plt.close('all')
	f, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True)
	ax1.imshow(IMG_X)
	ax1.set_title('Original Image')
	
	ax2.imshow(IMG_XX)
	ax2.set_title('Segmented Image')
	
	plt.xticks([])
	plt.yticks([])
	plt.savefig(r'results/'+text+'segmented_'+str(k)+'.jpg')
	plt.show()
	plt.close('all')

# ==================================================================

#X 				= randomData()
X,imR,imC,imD  	= imageData('img2.bmp')

linkage			= 'average'
metric			= 'euclidean'
k				= 2
mdl 			= hierarchical(X, n_clusters=k, linkage=linkage, metric=metric)



while True:
	try:
		### important ###
		# set whether labels  are used with the corresponding link coloring. 
		clust_k   = str(input('    Enter the number of clusters you want to view:'))
		clust_    = [ clust for clust in mdl.clusterings if clust['clusters_k'] == int(clust_k)]
		clust_x   = clust_[0]['clusters_x']
		clust_i   = clust_[0]['clusters_i']
		Z         = mdl.linkageMatrix
		
		y, C      = outputLC(X, clust_i)	
		
		colors      = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
		leaf_colors = {i: colors[lebel] for i, lebel in enumerate(y)}

		# plotting in case of using numerical data
		'''
		for i, cl in enumerate(clust_x):
			F1, F2 = zip(*cl)
			plt.scatter( F1, F2, c = [ colors[i] for _ in range(len(F1))  ] )
		plt.savefig(r'results/random_clustering_'+str(k)+'.jpg')
		plt.show()
		plt.close('all')
		'''
		
		# displaying image in case of using images
		clusteredImage = np.array([ C[i] for i in y ])
		k              = copy.deepcopy(clust_k)
		imageDisplay(X, clusteredImage, imR, imC, imD)

		# using labels y
		plotDendrogram(Z, labels=y)
		


	except ValueError:
		if clust_k == 'q': print("\nProgram is ended"); break
		print("Invalid number of clusters")




	
# ==================================================================
