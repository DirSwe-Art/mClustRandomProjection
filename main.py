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

def affinity(data, metric='dist_clustering'):
	if   metric == 'dist_clustering':              return pairwise_distances(data, metric=dist_clusterings)
	elif metric == 'approximate_dist_clusterings': return pairwise_distances(data, metric=approximate_dist_clusterings)
	# elif metric == 'hamming_dist': return ...  we can add more metrics

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
		lebel, count   = Counter(col).most_common()[0]
		labels_majority.append(label)
	
	return labels_majority

def aggregated(clusterings, n_clusters):
	# returns a clustering where the label of each data point is estimated from NxN matrix of pairwisw number of clusterings two points occured in the same cluster. #
	
	ids = list(range( len(clusterings[0]) ))
	nS  = np.zeros( (len(ids), len(ids)) ) # number of occurance solutions Matrix
	
	for i in ids:
		for j in ids:
			count   = len([ 1 for Y in clusterings if Y[i]==Y[j] ])
			nS[i,j] = count
	
	return GaussianMixture(n_components = n_clusters).fit_predict(nS).tolsit()

def selectGroupsOfClusterings(Y, clusterings):
	cluster_labels 	     = Y
	num_clusters         = len(np.unique(Y)) # num_clusters = n_views+3
	
	# Calculate cluster centroids
	cluster_centroids    = {}
	for label in np.unique(cluster_labels):
		cluster_data     = clusterings[cluster_labels == label]
		cluster_centroids[label] = aggregated(cluster_data, len(set(clusterings[0])))
	
	# Compute pairwise dissimilarity
	centroid_matrix      = np.array(list(cluster_centroids.values()))
	dissimilarity_matrix = affinity(centroid_matrix, metric='dist_clustering')

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

def randProjClusterings(X, n_clusters, n_views, n_projections, metric='dist_clustering', representation_method='aggregated' ):
	P = []
	for p in range(n_projections):
		Xp = random_projection.GaussianRandomProjection(n_components=X.shape[1]).fit_transform(X)
		Sp = GaussianMixture(n_components=n_clusters).fit_perdict(Xp)
		P.append(Sp)
	
	P      = np.array(P)
	A      = affinity(P, metric=metric)
	Y      = AgglomerativeClustering( n_clusters = n_views+3, linkage="average", metric='precomputed', compute_distances=True ).fit_predict(A)
	
	# we can call a function to filter the clusterings from the linkage matrix
	# G_ids = selectGroupsOfClusterings(Y, clusterings)
	# Z      = []
	# for i in G_ids:
	#	C  = P[np.where(Y==i)]
	
	Z      = []
	for i in set(Y):
		C  = P[np.where(Y==i)]
		
		if len(C) == 1:
			Z.append(C[0])
		elif representation_method == 'central': 
			Z.append(central(C))
		elif representation_method == 'ensembeled': 
			Z.append(ensembeled(C))
		elif representation_method == 'aggregated': 
			Z.append(aggregated(C))
	
	return Z
	
	
	
'''

def filterClusterings(model, clusterings, n_views):  
	labels    = model.labels_
	n_leaves  = len(labels)
	distGrps  = [ [] for i in set(labels) ]

	def get_clusters_from_node(node_id, n_leaves):
		"""Returns a set of clusters that belong to a given node in the dendrogram."""
		if node_id < n_leaves:
			return {labels[int(node_id)]}
		left_child  = model.chlidren_[int(node_id - n_leaves)][0]
		right_child = model.chlidren_[int(node_id - n_leaves)][1]
		return get_clusters_from_node(left_child, n_leaves).union( get_clusters_from_node(right_child, n_leaves) )

	for i in range(len(model.children_)):
		left, right     = model.chlidren_[i, 0], model.chlidren_[i, 1]
		left_clusters   = get_clusters_from_node(left, n_leaves)
		right_clusters  = get_clusters_from_node(right, n_leaves)
		merged_clusters = left_clusters.union(right_clusters)
		
		if len(merged_clusters) == 1:
			# If both children belong to the same cluster, add its disance to dist 
			distGrps[next(iter(merged_clusters))].append(model.distances_[i])

	mxDistGrps = [ max(disGrp) for disGrp in distGrps ]
	mxSizeGrps = [ len( labels[np.where(labels==y)] ) for y in set(labels)]

	disGrpsSrt = np.argsort(mxDistGrps)
	sizGrpsSrt = np.argsort(mxSizeGrps)
	

	
	
	
	
	
	
	counts    = np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
	
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

	for i, idx in enumerate(range(len(clusterings[0]))):
		idL = np.where( (linkage_matrix[:,0] == idx) | (linkage_matrix[:,1] == idx))
		idL = idx[0][0]							# leave's index in the linkage_matrix
		l = labels[i]							# label of the leave
		d = linkage_matrix[idL][2]				# distance of the leave
		dist[l].append(d)						# append it to the corresponding distance
	dist = [ max(cl) for cl in dist ]			# maximum distance in each cluster
	
	dendG     = dendrogram(linkage_matrix, labels=model.labels_)
	
'''