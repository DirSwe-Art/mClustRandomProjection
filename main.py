"""Multiple Clustering via Random Projection"""

# Authors: DirarSweidan
# License: DSB 3-Claus

from sklearn import random_projection
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt
import random, copy, sys, os
from scipy.sparse import lil_matrix
#import dask


# ========================================================================
def constructProjectionMatrix(d):
	# Returns a random projection transformation matrix spanned by a linearly independent (orthogonal) unit vectors. #
	
	bit_generator = np.random.PCG64DXSM()		# Create a 128-bit bit generator (PCG64DXSM)
	rng  = np.random.Generator(bit_generator)	# Create a Generator instance using the 128-bit bit generator
	A    = rng.normal(0, 1/d, size=(d,d))		# Generator's normal method to generate the matrix A
	#A    = np.random.normal(0, 1/d, size=(d,d)) # A random d x d matrix with entries from N(0, 1/d)
	Q, R = np.linalg.qr(A) 					   	# QR decomposition to A. (Q: orthogonal matrix, R: upper triangular matrix)
	M    = Q @ Q.T				             	# Prjection matrix (projects data onto a space spanned by the unit vectors in Q).
	return M

# enhanced function
def dist_clusterings(Ya, Yb):
    # Ensure inputs are numpy arrays
    Ya = np.array(Ya)
    Yb = np.array(Yb)
    
    # Create boolean masks for pairwise equality comparisons
    Ya_equal = np.equal.outer(Ya, Ya)  # Pairwise comparison of Ya
    Yb_equal = np.equal.outer(Yb, Yb)  # Pairwise comparison of Yb
    
    # XOR operation on the masks: True where only one is equal and the other isn't
    mismatch = np.triu(Ya_equal ^ Yb_equal, k=1)  # Only upper triangle to avoid double-counting
    
    # Count the number of mismatches
    d = np.sum(mismatch)
    
    return d

'''
def dist_clusterings(Ya, Yb):
    # Ensure the inputs are NumPy arrays for efficient operations
    Ya = np.array(Ya)
    Yb = np.array(Yb)
    
    # Total number of elements
    n = len(Ya)
    
    # Initialize distance counter
    d = 0

    # Loop over combinations of pairs (i, j) where i < j
    for i in range(n):
        for j in range(i+1, n):
            # XOR condition to check if the clustering mismatch happens
            if (Ya[i] == Ya[j]) ^ (Yb[i] == Yb[j]):
                d += 1
    return d
'''	
def approximate_dist_clusterings(Ya, Yb, th=1000):
	# Returns an approximate distance between two clustering solutions if the data size is larger than 100 points
	if len(Ya) < th: return dist_clusterings(Ya, Yb)
	
	ds_rand_Ys    = []
	for i in range(10):
		rand_ids = np.random.choice(range(len(Ya)), th, replace=False) # replace=False a value a is selected once.
		ds_rand_Ys.append( dist_clusterings([Ya[id] for id in rand_ids], [Yb[id] for id in rand_ids]) )
	return np.mean(ds_rand_Ys)

def affinity(data, affinity_metric='dist_clusterings'):
	if   affinity_metric == 'dist_clusterings':              return pairwise_distances(data, metric=dist_clusterings)
	elif affinity_metric == 'approximate_dist_clusterings': return pairwise_distances(data, metric=approximate_dist_clusterings)
	# elif affinity_metric == 'hamming_dist': return ...  we can add more metrics #

def central(clusterings):
	# returns a clustering from the pool that has the minimum sum of distnaces with all other clutserings. #
	
	A      = affinity(clusterings)
	id_min = np.argmin([ sum(A[row]) for row in A ])
	
	return clusterings[id_min]

def ensembeled(clusterings):
	# returns a clustering where the label of each data point is the majority voting of its labels among all clusterings. #
	
	zipped_clusterings = list(zip(*clusterings))
	
	labels_majority    = []
	for col in zipped_clusterings:
		label, count   = Counter(col).most_common()[0]
		labels_majority.append(label)
	
	return labels_majority


def aggregated(clusterings):
	# returns a clustering where the label of each data point is estimated from NxN matrix that containes 
	# the number of clusterings of each pairwise points where they belong to the same cluster. #

    ids = list(range(len(clusterings[0])))
    comb_ids = combinations(range(len(clusterings[0])), 2)

    nS = lil_matrix((len(ids), len(ids)))

    for i, j in comb_ids:  # Iterate over combinations lazily
        count = len([1 for Y in clusterings if Y[i] == Y[j]])
        if count > 0:  # Store non-zero values only
            nS[i, j] = count
            nS[j, i] = count  # Keep symmetry if needed

    nS = nS.tocsr()
    return GaussianMixture(n_components=len(set(clusterings[0]))).fit_predict(nS.toarray()).tolist()

def selectGroupsOfClusterings(Y, clusterings):
	# Returns the indices of clusterings that alternates groups with large sizes and large dissimilarities
	
	cluster_labels 	     = Y
	#num_clusters         = len(np.unique(Y)) # num_clusters = n_views+3
	
	# Calculate cluster centroids
	cluster_centroids    = {}
	for label in np.unique(cluster_labels):
		cluster_data     = clusterings[cluster_labels == label]
		cluster_centroids[label] = aggregated(cluster_data)
	
	# Compute pairwise dissimilarity
	centroid_matrix      = np.array(list(cluster_centroids.values()))
	dissimilarity_matrix = affinity(centroid_matrix, affinity_metric='dist_clustering')

	# Cluster sizes
	cluster_sizes        = {label: np.sum(cluster_labels == label) for label in np.unique(cluster_labels)}

	# Alternating iteration between largest and most dissimilar clusters
	selected_clusters 	 = []
	remaining_clusters   = set(cluster_sizes.keys())
	
	def select_largest_cluster():
		largest_cluster  = max(remaining_clusters, key=lambda label: cluster_sizes[label])
		return largest_cluster

	def select_most_dissimilar_cluster(last_selected):
		dissimilarities     = dissimilarity_matrix[last_selected, :]
		dissimilar_clusters = [(label, dissimilarities[label]) for label in remaining_clusters if label != last_selected and cluster_sizes[label] > 1]
		if dissimilar_clusters:  # Ensure there's a valid cluster to select
			most_dissimilar_cluster = max(dissimilar_clusters, key=lambda x: x[1])[0]
			return most_dissimilar_cluster
		else:
			return None  # Return None if no suitable cluster found
	
	# Alternating selection process
	is_largest = True
	while remaining_clusters:
		if is_largest:
			cluster = select_largest_cluster()
		else:
			if selected_clusters:
				last_selected = selected_clusters[-1]
				cluster = select_most_dissimilar_cluster(last_selected)
				if cluster is None:
					cluster = select_largest_cluster()  # Fallback if no suitable cluster found
			else:
				cluster = select_largest_cluster()  # fallback in case it's the first iteration
		
		selected_clusters.append(cluster)
		remaining_clusters.remove(cluster)
		is_largest = not is_largest  # Toggle between largest and most dissimilar

	# The `selected_clusters` list contains the order of cluster selections
	print("Order of selected clusters:", selected_clusters)
	return selected_clusters

def plotDendrogram(model, Y, resultsPath):
	# plots the dendrogram with various options concerning coloring labels and links. #
	def linkColorFunction(link_id):
		colors      = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
		n_leaves    = len(Y)
		link_colors = {}
		
		def get_clusters_from_node(node_id, n_leaves):
			"""Returns a set of clusters that belong to a given node in the dendrogram."""
			if node_id < n_leaves:
				return {Y[int(node_id)]}
			left_child  = int( Z[node_id - n_leaves][0] )
			right_child = int( Z[node_id - n_leaves][1] )
			return get_clusters_from_node(left_child, n_leaves).union( get_clusters_from_node(right_child, n_leaves) )
		
		for i in range(len(Z)):
			left, right     = int(Z[i, 0]), int(Z[i, 1])
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
	
	counts = np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
	for i, merge in enumerate(model.children_):
		current_count = 0
		for child_idx in merge:
			if child_idx < n_samples:
				current_count += 1  # leaf node
			else:
				current_count += counts[child_idx - n_samples]
		counts[i] = current_count

	Z = np.column_stack( [model.children_, model.distances_, counts] ).astype(float)
	
	plt.close('all')
	plt.figure(figsize=(10,6))
	
	Z2           = np.array(copy.deepcopy(Z))
	Z2[:, 2]     = [ float(i)/max(Z2[:, 2]) for i in Z2[:, 2] ]

	denZ = dendrogram( Z2,
				   leaf_rotation         = 0,
				   leaf_font_size        = 10,
				   labels				 = Y,
				   # customized link color based on give labels
				   color_threshold       = 0,
				   above_threshold_color = 'grey',
				   link_color_func       = linkColorFunction,

				   )
	# customized label color based on the given labels
	colors      = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	leaf_colors = {i: colors[lebel] for i, lebel in enumerate(Y)}
	ax          = plt.gca()
	x_labels    = ax.get_xmajorticklabels()
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
	plt.savefig(resultsPath+'dendrogram_'+str(datatype)+str(n_clusters)+'.jpg')
	plt.show()
	
def randProjClusterings(X, n_clusters=2, n_views=3, n_projections=30, dis_metric='dist_clusterings', clusterings_rep='aggregated' ):
	P = []
	for p in range(n_projections):
		XX = copy.deepcopy(X)
		#M  = constructProjectionMatrix(XX.shape[1])
		#Xp = XX @ M
		Xp = random_projection.GaussianRandomProjection(n_components=X.shape[1]).fit_transform(XX)
		Sp = GaussianMixture(n_components=n_clusters).fit_predict(Xp)
		P.append(Sp)
	
	P      = np.array(P)
	print('*** %d projections and clusterings are generated. ***'%(n_projections))
	
	A      = affinity(P, affinity_metric=dis_metric)
	print('*** Clusterings dissimilarity matrix is generated. ***')
	
	Y	   = AgglomerativeClustering( n_clusters=n_views, linkage="average", metric="precomputed", compute_distances=True ).fit_predict(A)
	print('*** Clusterings are groupped with an agglomeartive model. ***') 

	Z      = []
	for i in set(Y):
		C  = P[Y==i]
		
		if len(C) == 1:
			Z.append(C[0].tolist())
		elif clusterings_rep == 'central': 
			Z.append(central(C))
		elif clusterings_rep == 'ensembeled': 
			Z.append(ensembeled(C))
		elif clusterings_rep == 'aggregated': 
			Z.append(aggregated(C))
	
	print('*** Groups of similar clusterings are aggregated and represented. ***')
	return Z, AgglomerativeClustering( n_clusters=n_views, linkage="average", metric="precomputed", compute_distances=True ).fit(A)
	
# ====================================================================== #

def generate_data(type='432random'):
	if type == '432random':
		DATA = data432()
		k = 3
		n_views = 2
		datatype = '432random'
		return DATA, k, n_views, datatype
		
	elif type == '223random':
		DATA = data223()
		k = 2
		n_views = 2
		datatype = '223random'
		return DATA, k, n_views, datatype
		
	elif type == 'image':
		DATA, imRow, imCol, imDim = dataimg('source_images/img1.bmp')
		k = 2
		n_views = 12
		datatype = 'image'
		return DATA, k, n_views, datatype, imRow, imCol, imDim

def data223():								# 2 features, 2 clusters, 3 views
	X = []
	M = [[0,2], [0,4], [2.5,2], [2.5,4]]
	for m in M:
		X += np.random.multivariate_normal(m, np.identity(2)/3, size=300).tolist()
	
	X = np.array(X)
	X = X - X.mean(axis=0)					# center the data
	return X
	
def data432():								# 4 features, 3 clusters, 2 views
	X1 = []									# First view of the data (with F1, F2 features of each x)
	M  = [[2.5, 12.5], [9.5, 15], [16.5, 10.5]]			# Three centroids
	S  = [300, 300, 900]					# The first view's number of data points in each cluster
	
	for i in range(len(S)):
		X1 += np.random.multivariate_normal(M[i], np.identity(2)/3, size=S[i]).tolist()
	
	X2 = []									# Second view of the data (with F3, F4 features of each x)
	M  = [[5, 8], [6.5, 10.5], [8,8]]			# Three centroids
	S  = [600, 600, 300]					# The second view's number of data points in each cluster
	
	for i in range(len(S)):
		X2 += np.random.multivariate_normal(M[i], np.identity(2)/3, size=S[i]).tolist()
	
	X  = np.array([ X2[i] + X1[i] for i in range(len(X1)) ]) # Combine the two views (F1, F2, F3, F4)
	X  = X - X.mean(axis=0)					# Center the data (shift the data points toward the origine)
	return X

def dataimg(file):
	IMG = plt.imread(file)					# uint8 data type
	imRow, imCol, imDim = IMG.shape
	X = []
	
	for r in range(imRow):
		for c in range(imCol):
			X.append( IMG[r][c] )
			
	return np.array(X, dtype='uint8'), imRow, imCol, imDim

def plot_clusters(DATA, colors, t, resultsPath):
	if datatype=='223random' or datatype=='432random': return random_clusters(DATA, colors, t, resultsPath)
	if datatype == 'image': return image_clusters(DATA, colors, t, resultsPath) 
	
def random_clusters(DATA, colors, t, resultsPath):
	fig, (ax1a, ax2a) = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
	
	ax1a.scatter( *np.array([ *zip(*DATA) ])[:2], c=colors, marker='+' )
	ax1a.set_title('Original Space')
	ax1a.set_xlabel('Feature 1')
	ax1a.set_ylabel('Feature 2')
	
	if len(DATA[0]) > 2:
		ax2a.scatter( *np.array([ *zip(*DATA) ])[2:4], c=colors, marker='+' )
		ax2a.set_title('Original Space')
		ax2a.set_xlabel('Feature 3')
		ax2a.set_ylabel('Feature 4')
	
	if len(DATA[0]) > 2:
		plt.savefig(resultsPath+'random_3_clustering_n_'+str(t)+'.png')
	else:
		plt.savefig(resultsPath+'random_2_clustering_n_'+str(t)+'.png')
	
	plt.close('all')	

def image_clusters(DATA, colors, t, resultsPath):
	IMG_DATA = copy.deepcopy(DATA).reshape(imRow, imCol, imDim)
	
	f, (ax1, ax3) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 5))
	ax1.imshow(IMG_DATA)
	ax1.set_title('Source image')								# view the original space
	
	ax3.imshow(np.array(colors, dtype='uint8').reshape(imRow, imCol, imDim)) 	# view the segmented space (center colors)
	ax3.set_title('Clustering Solution')
	
	plt.savefig(resultsPath+'image_clustering_n_'+str(t)+'.png')
	plt.close('all')	
	
# ====================================================================== #


resultsPath     = r'C:/ExperimentalResults/Results/results_MultipleClusteringsViaRandomProjection/'
if not os.path.exists(resultsPath): os.makedirs(resultsPath)



(DATA, n_clusters, 
 n_views, datatype,   
# imRow, imCol, imDim)= generate_data(type= 'image')		# 'image'
 					 )= generate_data(type= '432random')	# '432random', '223random'

n_projections 		 = 60
dis_metric			 = 'dist_clusterings'				# 'dist_clusterings', 'approximate_dist_clusterings'
clusterings_rep 	 = 'aggregated'						# 'centeral', 'ensembeled', 'aggregated'



clust_arr, clust_mdl = randProjClusterings(
						DATA, 
						n_clusters 	    = n_clusters, 
						n_views 	    = 4,
						n_projections   = n_projections, 
						dis_metric 	    = dis_metric, 
						clusterings_rep = clusterings_rep ) 


#plotDendrogram(clust_mdl, clust_mdl.labels_, resultsPath)


for clust_id, labels in enumerate(clust_arr):
	if datatype == 'image':
		# Coloring RGB pixels with thier cluster correspondiing color (2 colors, 1 for each cluster)
		clr = [ [0, 0, 0], [255, 255, 255] ] 	 
		
		# coloring RGB pixels with their cluster means
		#clr = [ [np.mean(col) for col in zip(*DATA[labels==cl])] for cl in set(labels) ] 
	else:
		# Coloring data points with thier cluster correspondiing color
		clr = ['brown', 'green', 'black' ,'cornflowerblue', 'yellow', 'orange']
	
	plot_clusters( DATA, [ clr[i] for i in labels ], clust_id, resultsPath )
	
