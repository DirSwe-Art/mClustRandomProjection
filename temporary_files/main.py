"""Multiple Clustering via Random Projection"""

# Authors: DirarSweidan
# License: DSB 3-Claus

from sklearn import random_projection
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, cut_tree
import numpy as np
import matplotlib.pyplot as plt
import random, copy, math, time, datetime, os, sys, gc, h5py
from scipy.sparse import csc_array, csr_array, lil_array
import pandas as pd
#import dask


# ========================================================================
def constructProjectionMatrix(d):
	# Returns a random projection transformation matrix spanned by a linearly independent (orthogonal) unit vectors. #
	
	bit_generator = np.random.PCG64DXSM()		# Create a 128-bit bit generator (PCG64DXSM)
	rng  = np.random.Generator(bit_generator)	# Create a Generator instance using the 128-bit bit generator
	A    = rng.normal(0, 1/math.sqrt(d), size=(d,d))		# Generator's normal method to generate the matrix A
	M    = A @ np.linalg.inv(A.T @ A) @ A.T		# Prjection matrix (projects data onto a space spanned by the unit vectors in A).#

	return A

# enhanced function
def dist_clusterings(Ya, Yb):
    Ya = np.array(Ya)							# Ensure inputs are numpy arrays
    Yb = np.array(Yb)
    
    # Create boolean masks for pairwise equality comparisons
    Ya_equal = np.equal.outer(Ya, Ya)  			# Pairwise comparison of Ya
    Yb_equal = np.equal.outer(Yb, Yb)  			# Pairwise comparison of Yb
    
    # XOR operation on the masks: True where only one is equal and the other isn't
    mismatch = np.triu(Ya_equal ^ Yb_equal, k=1)# Only upper triangle to avoid double-counting
    
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

def approximate_dist_clusterings(Ya, Yb, th=2200):
	# Returns an approximate distance between two clustering solutions if the data size is larger than 100 points
	if len(Ya) < th: return dist_clusterings(Ya, Yb)
	
	ds_rand_Ys    = []
	for i in range(10):
		rand_ids = np.random.choice(range(len(Ya)), th, replace=False) # replace=False a value a is selected once.
		ds_rand_Ys.append( dist_clusterings([Ya[id] for id in rand_ids], [Yb[id] for id in rand_ids]) )
	return np.mean(ds_rand_Ys)

def affinity(data, affinity_metric='dist_clusterings'):
	if   affinity_metric == 'dist_clusterings':             return pairwise_distances(data, metric=dist_clusterings)
	elif affinity_metric == 'approximate_dist_clusterings': return pairwise_distances(data, metric=approximate_dist_clusterings)
	# elif affinity_metric == 'hamming_dist': return ...  we can add more metrics #

def central(clusterings):
	# returns a clustering from the pool that has the minimum sum of distnaces with all other clutserings. #
	
	A      = affinity(clusterings)
	id_min = np.argmin([ sum(A[row]) for row in A ])
	
	return clusterings[id_min]

def ensemble(clusterings):
	# returns a clustering where the label of each data point is the majority voting of its labels among all clusterings. #
	
	zipped_clusterings = list(zip(*clusterings))
	
	labels_majority    = []
	for col in zipped_clusterings:
		label, count   = Counter(col).most_common()[0]
		labels_majority.append(label)
	
	return labels_majority

'''
def aggregate(clusterings):
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


# better performance with enough memory
def aggregate(G):
	# dictionary for sample pairwise equality comparisons in each clustering
	dict_ = {}
	for s_id, S in enumerate(G):
		dict_[s_id]= np.equal.outer(S,S);print(s_id,'EqualOuter')

	xS = [] # xi and each xj are together (m-element row for each solution)
	xC = [] # xi representation (the sum of xj-column over all solutions where xi,xj are together
    
	print('len G0',len(G[0]))
	for x_id in range(len(G[0])):
		for s_id, S in enumerate(G):
			xS.append(dict_[s_id][x_id])
		sums  = np.sum(np.array(xS), axis=0)
		xC.append(sums)
		xS    = []
		print('x', x_id)
	print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')
		
	return GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist()

def aggregate(G):
	# dictionary for sample pairwise equality comparisons in each clustering
	dict_ = {}
	for s_id, S in enumerate(G):
		dict_[s_id]= np.equal.outer(S,S);print(s_id,'EqualOuter')
	
	xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
	xC = pd.DataFrame({}, index=range(len(G[0])), columns=range(len(G[0])))
    
	print('len G0',len(G[0]))
	for x_id in range(len(G[0])):
		for s_id, S in enumerate(G):
			xS.iloc[s_id] = dict_[s_id][x_id]
		sums  = xS.sum(axis=0)
		xC.iloc[x_id] = sums
		xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
		print('x', x_id)
	print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')
		
	return GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist()


# does not work
def aggregate(G):
	for s_id, S in enumerate(G):
		exec('global dict_'+str(s_id))
		globals()['dict_G_S'+str(s_id)]={}
		globals()['dict_G_S'+str(s_id)][s_id] = np.equal.outer(S,S);print(s_id,'EqualOuter')
	
	xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
	xC = pd.DataFrame({}, index=range(len(G[0])), columns=range(len(G[0])))
	    
	print('len G0',len(G[0]))
	for x_id in range(len(G[0])):
		for s_id, S in enumerate(G):
			xS.iloc[s_id] = globals()['dict_G_S'+str(s_id)][s_id][x_id]
		sums  = xS.sum(axis=0)
		xC.iloc[x_id] = sums
		xS = pd.DataFrame({}, index=range(len(G)), columns=range(len(G[0])))
		print('x', x_id)
	
	print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')

	return GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist()
'''

'''
# Function to fit a model in batches
def batch_fit_gmm_with_min_passes(gmm_model, X, batch_size, min_passes=1, tol=1e-4, max_no_improvement=10):
    n_samples = X.shape[0]
    prev_log_likelihood = None
    no_improvement = 0
    num_passes = 0

    while num_passes < min_passes or no_improvement < max_no_improvement:
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            gmm_model.fit(X_batch)
            
            # Get log-likelihood of the model on the current batch
            current_log_likelihood = gmm_model.score(X_batch)
            
            # Check for convergence
            if prev_log_likelihood is not None:
                change_in_likelihood = abs(prev_log_likelihood - current_log_likelihood)
                if change_in_likelihood < tol:
                    no_improvement += 1
                else:
                    no_improvement = 0
                    
                # Stop if no improvement for max_no_improvement batches and all passes are completed
                if no_improvement >= max_no_improvement and num_passes >= min_passes:
                    print(f"Converged after {num_passes} full passes and {i // batch_size} batches.")
                    return gmm_model
            
            prev_log_likelihood = current_log_likelihood
        
        num_passes += 1
    
    return gmm_model

# Function to predict in batches
def batch_predict_gmm(gmm_model, X, batch_size):
    n_samples = X.shape[0]
    predictions = []
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        predictions.append(gmm_model.predict(X_batch))  # For cluster labels
    return np.concatenate(predictions)
'''

def batch_fit_kmeans(kmeans_model, X, batch_size):
    n_samples = X.shape[0]; print('\n\t -> Fitting ...' )
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]; print('\t  - Batch:', i+batch_size )
        kmeans_model.partial_fit(X_batch)  # Incrementally fit the mini-batch
    return kmeans_model


def batch_predict(kmeans_model, X, batch_size):
    n_samples = X.shape[0]; print('\n\t -> Predicting ...' )
    predictions = []
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]; print('\t  - Batch:', i+batch_size )
        predictions.append(kmeans_model.predict(X_batch))
    return np.concatenate(predictions)

def aggregate(G, label):
	def pairwiseOccurance(M1_path, M2_path, result_path, shape, chunk_size=10000):
	    #Perform np.equal.outer(M1, M2) using np.memmap to handle large boolean arrays.
	
	    #Parameters:
	    #- M1_path: str, path to memory-mapped file for M1
	    #- M2_path: str, path to memory-mapped file for M2
	    #- result_path: str, path to memory-mapped file for the result
	    #- shape_M1: tuple, shape of M1 (must be 1D)
	    #- shape_M2: tuple, shape of M2 (must be 1D)
	    #- chunk_size: int, the size of chunks to process in memory
	    
	    #Result is stored in `result_path` as a memory-mapped file.
	    
		mmap_M1 = np.memmap(M1_path, dtype=np.bool_, mode='r', shape=shape)
		mmap_M2 = np.memmap(M2_path, dtype=np.bool_, mode='r', shape=shape)
	    
	    # Prepare a memory-mapped file for the result
		result_shape = (shape[0], shape[0])
		mmap_result  = np.memmap(result_path, dtype=np.bool_, mode='w+', shape=result_shape)
	
	    # Process in chunks to handle memory efficiently
		for i in range(0, shape[0], chunk_size):
			chunk_end = min(i + chunk_size, shape[0])
	        
	        # Perform the np.equal.outer operation for the current chunk
			mmap_result[i:chunk_end, :] = np.equal.outer(mmap_M1[i:chunk_end], mmap_M2[:])
	        
	        # Flush changes to disk after each chunk
			mmap_result.flush()
	
	    # Delete memory-mapped arrays to free up RAM
		del mmap_M1
		del mmap_M2
		del mmap_result
	
	    # Force garbage collection to ensure memory is freed
		gc.collect()
	
	    # Optionally return the result path
		return result_path

	def allPairwiseOccurance(G, result_path):
		# Returns one dictionary saved externally for all solutions in G.  
		# Each key is for one solution's elements pairwise equality comparison.
		with h5py.File(result_path, 'w') as hf:
			for s_id, S in enumerate(G):
				np.memmap('S'+str(s_id)+'_1', dtype=np.int8, mode='w+', shape=S.shape)[:] = S[:] # => 's_id S1.dat'
				np.memmap('S'+str(s_id)+'_2', dtype=np.int8, mode='w+', shape=S.shape)[:] = S[:] # => 's_id S2.dat'
				SS_result_path = pairwiseOccurance('S'+str(s_id)+'_1', 'S'+str(s_id)+'_2', 'S'+str(s_id)+'_result', S.shape, chunk_size=10000) #  # => 'result.dat'
				
				dset = hf.create_dataset('S'+str(s_id), shape=(len(S), len(S)), dtype=np.bool_) # => s_id (m,m) dataset

				# Open the memmap file in read mode
				SS_result_data = np.memmap(SS_result_path, dtype=np.bool_, mode='r', shape=(len(S), len(S)))

				# Process the data in chunks and write to the HDF5 file incrementally
				for i in range(0, S.shape[0], 10000):
					chunk_end = min(i + 10000, S.shape[0])
					dset[i:chunk_end] = SS_result_data[i:chunk_end]  # Write chunk directly to the HDF5 dataset
				
				del SS_result_data
				time.sleep(10)
				os.remove('S'+str(s_id)+'_1')
				os.remove('S'+str(s_id)+'_2')
				os.remove(SS_result_path)
				time.sleep(10)
				
				
				
				# After this loop, the entire memmap data will be written to the HDF5 file
			
				print('\t -> Pairwise occurancies in solution %d'%s_id)
		return result_path

	def occuranceFreq(x_id, G, G_dict_path):
		# Returns a vector representation of one data point x where frequencies  
		# of which it occures together with other points across solutions in G.
		with h5py.File(G_dict_path, 'r') as hf:
			xS = pd.DataFrame( {}, columns=range(len(G)), dtype=np.int8)
			for s_id, S in enumerate(G):
				xS[str(s_id)] = hf['S'+str(s_id)][x_id]
			return xS.sum(axis=1)
	
	G_dict_path = allPairwiseOccurance(G, 'G_dict') # => G_dict.h5 
	
	# DictMethod
	#xC    = pd.DataFrame( {}, columns=range(len(G[0])) , dtype=np.int8) 	# Matrix representation for all points according to G
	
	
	# MemMethod
	xC     = np.memmap('G_matrix', dtype=np.int8, mode='w+', shape=(len(G[0]),len(G[0])))
	
	print('\n\t -> Occurance frequencies X^C (%d, %d) for all X(xi, xj) in all G(S). Save in an external file.'%(len(G[0]),len(G[0])) )
	for x_id in range(len(G[0])):
		freq 		= occuranceFreq(x_id, G, G_dict_path)
		
		# DictMethod
		#xC[x_id]	= freq                                             	 	 
		
		# MemMethod
		xC[x_id, :]	= freq
		
		if x_id % 10000 == 0: print('\t -> batch', x_id)
	
	# MemMethod
	xC.flush(); time.sleep(10); del xC ; time.sleep(5)
														
	print('\n\t -> Emptying the RAM, reading the external X^C file')
	
	# MemMethod
	xC_memory   = np.memmap('G_matrix', dtype=np.int8, mode='r', shape=(len(G[0]),len(G[0])))
	kmeans      = MiniBatchKMeans(n_clusters=len(set(G[0])), batch_size=10000, max_iter=100, tol=1e-4, max_no_improvement=15, random_state=42)
	kmeans      = batch_fit_kmeans(kmeans, xC_memory, batch_size=10000)
	predictions = batch_predict(   kmeans, xC_memory, batch_size=10000)
	del xC_memory
	
	time.sleep(10)
	os.remove(G_dict_path)
	os.remove('G_matrix')
	time.sleep(10)
	

	#return GaussianMixture(n_components=len(set(G[0]))).fit_predict(xC).tolist()
	return predictions

def selectGroupsOfClusterings(Y, clusterings):
	# Returns the indices of clusterings that alternates groups with large sizes and large dissimilarities
	
	cluster_labels 	     = Y
	#num_clusters         = len(np.unique(Y)) # num_clusters = n_views+3
	
	# Calculate cluster centroids
	cluster_centroids    = {}
	for label in np.unique(cluster_labels):
		cluster_data     = clusterings[cluster_labels == label]
		cluster_centroids[label] = aggregate(cluster_data)
	
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

def computeLinkageFromModel(model):
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
	return Z

def plotDendrogram(model, Y, resultsPath):
	# plots the dendrogram with various options concerning coloring labels and links. #
	def linkColorFunction(link_id):
		colors      = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19' ]
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
	
	Z = computeLinkageFromModel(model)
	
	plt.close('all')
	plt.figure(figsize=(10,6))
	
	Z2           = np.array(copy.deepcopy(Z))
	#Z2[:, 2]     = [ float(i)/max(Z2[:, 2]) for i in Z2[:, 2] ]

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
	colors      = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19' ]
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
	
	plt.title('Dendrogram with Cluster-based Link and Leaf Colors - '+str(data_name[6:]))
	plt.xlabel('Sample index')
	plt.ylabel('Distance')
	plt.savefig(resultsPath+'dendrogram_'+data_name[6:]+'_k_'+str(n_clusters)+'.jpg')
	#plt.show()

def large_labels_first(DATA, Y):
	DATA         = np.array(copy.deepcopy(DATA))
	Y            = np.array(Y)
	
	# The size of each label
	label_sizes  = [ len(DATA[Y==y]) for y in set(Y) ]
	
	# Sort label indices large to small
	label_sorted = np.flip(np.argsort(label_sizes))

	# assign new labels
	new_labels   = np.zeros(Y.shape, dtype=int)
	for new_lable, old_label in enumerate(label_sorted):
		new_labels[Y==old_label] = new_lable
	
	return new_labels
	
def mClustRandomProjection(X, n_projections=60, n_clusters=2, dis_metric='dist_clusterings'):
	print('\n*** The algorithm is started with the following parameter values: \n    %d projections, each with %d clusters.\n'%(n_projections, n_clusters))
	
	P = []
	for p in range(n_projections):
		XX = copy.deepcopy(X)
		Mp = constructProjectionMatrix(XX.shape[1])
		Xp = XX @ Mp
		#Xp = random_projection.GaussianRandomProjection(n_components=X.shape[1]).fit_transform(XX)
		Sp = GaussianMixture(n_components=n_clusters).fit_predict(Xp)
		P.append(Sp)
	
	P      = np.array(P)
	print('*** %d projections and clusterings are generated. ***'%(n_projections))
	
	A      = affinity(P, affinity_metric=dis_metric)
	print('*** Clusterings dissimilarity matrix is generated. ***')
	
	M      = AgglomerativeClustering( linkage="average", metric="precomputed", compute_distances=True ).fit(A)
	print('*** Clusterings hierarchy is generated with an agglomeartive model. ***') 

	return M, P

def representative_solutions(model, clusterings, n_views=3, clusterings_rep='aggregate'):
	print('*** Grouping similar clusterings is started. ***')
	R      = []
	Z      = computeLinkageFromModel(model)
	G      = np.array(cut_tree(Z, n_clusters=n_views).flatten())
	
	plotDendrogram(model, G, resultsPath)
	
	print('*** Aggregating groups of clusterings is started. ***')
	for label in set(G):
		C  = clusterings[G==label]
		
		if len(C) == 1:
			print('*** Group (%d) has one clustering solution. No aggregation is needed. ***'%label)
			R.append(C[0].tolist())
		elif clusterings_rep == 'central': 
			print('*** Finding the central solution of group (%d). ***'%label)
			R.append(central(C))
		elif clusterings_rep == 'ensemble': 
			print('*** Computing the ensemble solution of group (%d). ***'%label)
			R.append(ensemble(C))
		elif clusterings_rep == 'aggregate': 
			print('*** Aggregating clusterings of group (%d). ***'%label)
			R.append(aggregate(C, label))
	
	print('*** Groups of clusterings are aggregated to representative clusterings. ***')
	return np.array(R)
	
# ====================================================================== #

def generate_data(data_name='random432', format='text'):
	if data_name == 'random432':
		DATA = data432()
		k = 3
		return DATA, k, data_name
		
	elif data_name == 'random223':
		DATA = data223()
		k = 2
		return DATA, k, data_name
		
	elif data_name[0:5] == 'image':
		DATA, imRow, imCol, imDim = dataimg('source_images/'+data_name, format=format)
		k = 2
		return DATA, k, data_name, imRow, imCol, imDim

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

def dataimg(file, format='bmp'):
	print('*** Reading (',file,') image data ... ', end='')
	
	IMG = plt.imread(file, format=format)					# uint8 data type
	if IMG.shape[-1] == 4:
		print('the image has 4 dimesions. Dropping the alpha channel ... ', end='')
		IMG = IMG[..., :3]  				# Converting RGBA to RGB. Drop the alpha channel
	
	imRow, imCol, imDim = IMG.shape
	X = []
	
	for r in range(imRow):
		for c in range(imCol):
			X.append( IMG[r][c] )
	
	print('The shape is %d rows, %d columns, %d dimensions ***'%(imRow, imCol, imDim))
	return np.array(X, dtype='uint8'), imRow, imCol, imDim

def plot_clusters(DATA, colors, t, resultsPath):
	if data_name      =='random223' or data_name=='random432': return random_clusters(DATA, colors, t, resultsPath)
	if data_name[0:5] == 'image': return image_clusters(DATA, colors, t, resultsPath) 
	
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

	# If the input colors array has 4 channels (RGBA), convert it to RGB
	if np.array(colors).shape[-1] == 4:
		colors = np.array(colors)[..., :3]  # Drop the alpha channel if present
	reshaped_colors = np.array(colors, dtype='uint8').reshape(imRow, imCol, imDim)

	# Plotting the original and clustered image
	f, (ax1, ax3) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 5))
	
	# Display the original image
	ax1.imshow(IMG_DATA)
	ax1.set_title('Source image')
	
	# Display the clustered image
	ax3.imshow(reshaped_colors)
	ax3.set_title('Clustering Solution')
	
	# Display the clustered image
	plt.savefig(resultsPath+data_name+'_'+str(t)+'.png')
	plt.close('all')	
	
# ====================================================================== #

starting_time   = time.time()
resultsPath     = r'C:/ExperimentalResults/Results/results_MultipleClusteringsViaRandomProjection/'
if not os.path.exists(resultsPath): os.makedirs(resultsPath)

## Generate Data
(DATA, n_clusters, 
 data_name,   
 imRow, imCol, imDim)= generate_data(data_name= 'image_x-ray-chest-small.bmp', format='bmp')	# 'image1.png', 'image2.png', 'image3.png', 'image4.png', 'image_x-ray.bmp'
# 					 )= generate_data(data_name= '223random')	# '432random', '223random'


## Settings
n_projections 		 = 60
n_clusters           = 7
n_views              = 3
dis_metric			 = 'approximate_dist_clusterings'		# 'dist_clusterings', 'approximate_dist_clusterings'
clusterings_rep 	 = 'aggregate'							# 'centeral', 'ensemble', 'aggregate'

M_mdl, P 	   		 = mClustRandomProjection(
						DATA, 
						n_projections   = n_projections, 
						n_clusters 	    = n_clusters, 
						dis_metric 	    = dis_metric )

plotDendrogram(M_mdl, M_mdl.labels_, resultsPath)
print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')

n_views   = 7
starting_time   = time.time()

representatives 	 = representative_solutions(
						model           = M_mdl,
						clusterings     = P,
						n_views 	    = int(n_views), 
						clusterings_rep = clusterings_rep ) 

for S_id, S in enumerate(representatives):
	if data_name[0:5] == 'image':
		# Coloring RGB pixels with thier cluster correspondiing color (2 colors, 1 for each cluster)
		#clr = [ [0, 0, 0], [255, 255, 255] ] 	 
		clr = [[0,0,0], [0,128,0], [255,140,0], [165,42,42], [255,255,255], [65,105,225], [255,215,0] ]
		
		
		# coloring RGB pixels with their cluster means
		#clr = [ [np.mean(col) for col in zip(*DATA[S==cl])] for cl in set(S) ] 
	else:
		# Coloring data points with thier cluster correspondiing color
		clr = ['black', 'green', 'orange', 'brown', 'white', 'cornflowerblue', 'yellow', ]
	
	sorted_labels = large_labels_first(DATA, S) 
	plot_clusters( DATA, [ clr[i] for i in sorted_labels ], S_id, resultsPath )

print('*** Final solutions are presented. ***')
		
'''
while True:
	try:
		### important ###
		# set whether labels  are used with the corresponding link coloring. 
		n_views   = str(input('    Enter the number of number of views of clustering solutions: '))
		starting_time   = time.time()
		
		representatives 	 = representative_solutions(
								model           = M_mdl,
								clusterings     = P,
								n_views 	    = int(n_views), 
								clusterings_rep = clusterings_rep ) 
		
		for S_id, S in enumerate(representatives):
			if data_name[0:5] == 'image':
				# Coloring RGB pixels with thier cluster correspondiing color (2 colors, 1 for each cluster)
				#clr = [ [0, 0, 0], [255, 255, 255] ] 	 
				clr = [[0,0,0], [0,128,0], [255,140,0], [165,42,42], [255,255,255], [65,105,225], [255,215,0] ]
				
				
				# coloring RGB pixels with their cluster means
				#clr = [ [np.mean(col) for col in zip(*DATA[S==cl])] for cl in set(S) ] 
			else:
				# Coloring data points with thier cluster correspondiing color
				clr = ['black', 'green', 'orange', 'brown', 'white', 'cornflowerblue', 'yellow', ]
			
			sorted_labels = large_labels_first(DATA, S) 
			plot_clusters( DATA, [ clr[i] for i in sorted_labels ], S_id, resultsPath )
		
		print('*** Final solutions are presented. ***')
			

	except ValueError:
		if n_views == 'q': print("\nProgram is ended"); break
		print("Invalid number of clusters")
'''




########################################


### test functions
'''
def aggregate(G):
	# Returns an aggregated representative solution for all solutions in G.
	# This function partially solves the memory lack issue with large datasets. 
	
	def pairwiseOccurance(M1, M2):
		# Returns an (m,m) matrix for (M1,M2) pairwise equaliity comparisons. 
		return np.equal.outer(M1,M2)

	def allOccurance(G):
		# Returns one dictionary for all solutions in G. Each key is for 
		# one solution's elements pairwise equality comparison.
		dict_ = {}
		for s_id, S in enumerate(G):
			dict_[s_id]= pairwiseOccurance(S,S)
			#print('\t -> Pairwise occurancies in solution %d'%s_id)
		return dict_
	
	def occuranceFrequencies(x_id, G, dict_):
		# Returns a vector representation of one data point x where frequencies  
		# of which it occures together with other points across solutions in G.  
		xS = pd.DataFrame( {}, columns=range(len(G)), dtype=np.int8)
		for s_id, S in enumerate(G):
			xS[s_id] = dict_[s_id][x_id]
		return xS.sum(axis=1)

	def batch_fit_kmeans(kmeans_model, X, batch_size=10000):
		# Returns a batch fitted k-means model for a large matrix
	    n_samples = X.shape[0] #; print('\n\t -> Fitting ...' )
	    for i in range(0, n_samples, batch_size):
	        X_batch = X[i:i+batch_size] #; print('\t  - Batch:', i+batch_size )
	        kmeans_model.partial_fit(X_batch)  # Incrementally fit the mini-batch
	    return kmeans_model

	def batch_predict(kmeans_model, X, batch_size=10000):
		# Returns a batch predicted cluster labels for a large matrix
	    n_samples = X.shape[0] #; print('\n\t -> Predicting ...' )
	    predictions = []
	    for i in range(0, n_samples, batch_size):
	        X_batch = X[i:i+batch_size] #; print('\t  - Batch:', i+batch_size )
	        predictions.append(kmeans_model.predict(X_batch))
	    return np.concatenate(predictions)
	
	dict_ = allOccurance(G)
	xC    = pd.DataFrame( {}, columns=range(len(G[0])) , dtype=np.int8) 	# Matrix representation for all points according to G
	for x_id in range(len(G[0])):
		freq 		= occuranceFrequencies(x_id, G, dict_)
		xC[x_id]	= freq	
	del dict_														

	kmeans = MiniBatchKMeans(n_clusters=len(set(G[0])), batch_size=10000,  max_iter=100, tol=1e-4,  max_no_improvement=10, random_state=42)
	kmeans = batch_fit_kmeans(kmeans, xC, batch_size=10000)
	predictions = batch_predict(kmeans, xC, batch_size= 10000)

	return predictions
'''

'''
def aggregate(G, label):
	# Returns an aggregated representative solution for all solutions in G.
	# This function solves the memory lack issue with large data sets. 
	# However, it is effective but not efficient, meaning that it takes a long
	# processing time since operations are not accomplished through a virtual 
	# memory on the hard disk.
	
	def pairwiseOccurance(M1, M2):
		# Returns an (m,m) matrix for (M1,M2) pairwise equaliity comparisons. 
		return np.equal.outer(M1,M2)

	def allPairwiseOccurance(G):
		# Returns one dictionary for all solutions in G. Each key is for 
		# one solution's elements pairwise equality comparison.
		dict_ = {}
		for s_id, S in enumerate(G):
			dict_[s_id]= pairwiseOccurance(S,S)
			print('\t -> Pairwise occurancies in solution %d'%s_id)
		return dict_
	
	def occuranceFrequencies(x_id, G, dict_):
		# Returns a vector representation of one data point x where frequencies  
		# of which it occures together with other points across solutions in G.  
		xS = pd.DataFrame( {}, columns=range(len(G)), dtype=np.int8)
		for s_id, S in enumerate(G):
			xS[s_id] = dict_[s_id][x_id]
		return xS.sum(axis=1)

	def batch_fit_kmeans(kmeans_model, X, batch_size=10000):
		# Returns a batch fitted k-means model for a large matrix
	    n_samples = X.shape[0]; print('\n\t -> Fitting ...' )
	    for i in range(0, n_samples, batch_size):
	        X_batch = X[i:i+batch_size]; print('\t  - Batch:', i+batch_size )
	        kmeans_model.partial_fit(X_batch)  # Incrementally fit the mini-batch
	    return kmeans_model

	def batch_predict(kmeans_model, X, batch_size=10000):
		# Returns a batch predicted cluster labels for a large matrix
	    n_samples = X.shape[0]; print('\n\t -> Predicting ...' )
	    predictions = []
	    for i in range(0, n_samples, batch_size):
	        X_batch = X[i:i+batch_size]; print('\t  - Batch:', i+batch_size )
	        predictions.append(kmeans_model.predict(X_batch))
	    return np.concatenate(predictions)
	
	dict_ = allPairwiseOccurance(G)
	
	if os.path.exists(str(label)+'_matrix.dat'): 
		os.remove(str(label)+'_matrix.dat')
		time.sleep(5)
	
	xC     = np.memmap(str(label)+'_matrix.dat', dtype=np.int8, mode='w+', shape=(len(G[0]),len(G[0])))
	
	print('\n\t -> Occurance frequencies X^C (%d, %d) for all X(xi, xj) in all G(S). Save in an external file.'%(len(G[0]),len(G[0])) )
	for x_id in range(len(G[0])):
		freq 		= occuranceFrequencies(x_id, G, dict_)
		xC[x_id, :]	= freq
		#if x_id % 10000 == 0: print('\t -> batch', x_id)
	
	xC.flush(); time.sleep(10); del xC ; time.sleep(5)
	
	del dict_ ; time.sleep(5)  														
	
	print('\n\t -> Emptying the RAM, reading the external X^C file')
	
	xC_memory   = np.memmap(str(label)+'_matrix.dat', dtype=np.int8, mode='r', shape=(len(G[0]),len(G[0])))
	kmeans      = MiniBatchKMeans(n_clusters=len(set(G[0])), batch_size=10000, max_iter=100, tol=1e-4, max_no_improvement=15, random_state=42)
	kmeans      = batch_fit_kmeans(kmeans, xC_memory, batch_size=10000)
	predictions = batch_predict(   kmeans, xC_memory, batch_size=10000)
	del xC_memory; time.sleep(5)
	
	return predictions
'''



'''

def selectGroupsOfClusterings(Y, clusterings):
	# Returns the indices of clusterings that alternates groups with large sizes and large dissimilarities
	
	cluster_labels 	     = Y
	#num_clusters         = len(np.unique(Y)) # num_clusters = n_views+3
	
	# Calculate cluster centroids
	cluster_centroids    = {}
	for label in np.unique(cluster_labels):
		cluster_data     = clusterings[cluster_labels == label]
		cluster_centroids[label] = aggregate(cluster_data)
	
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
'''

'''
def geterate_clusterings_pool(X, n_projections=60, n_clusters=2, dis_metric='dist_clusterings'):
	print('\n*** The algorithm is started with the following parameter values: \n    %d projections, each with %d clusters.\n'%(n_projections, n_clusters))
	
	P = []
	for p in range(n_projections):
		Mp = construct_projection_matrix(X.shape[1])
		Xp = X @ Mp
		#Xp = random_projection.GaussianRandomProjection(n_components=X.shape[1]).fit_transform(X)
		Sp = GaussianMixture(n_components=n_clusters).fit_predict(Xp)
		P.append(Sp)
	
	P      = np.array(P)
	print('*** %d projections and clusterings are generated. ***'%(n_projections))
	
	return P

def clusterings_hierarchy(pool, dis_metric='dist_clusterings'):	
	A      = affinity(pool, affinity_metric=dis_metric)
	print('*** Clusterings dissimilarity matrix is generated. ***')
	
	M      = AgglomerativeClustering( linkage="average", metric="precomputed", compute_distances=True ).fit(A)
	print('*** Clusterings hierarchy is generated with an agglomeartive model. ***') 

	D      = personalized_dendrogram(M, [o for _ in M.labels_])
	
	plot_dendrogram(D, [0 for _ in M.labels_])
	print('*** Clusterings hierarchy is plotted. Please analyze it. ***') 
	
	return M, A


def label_clusterings__(model, resultsPath=None):
	try:
		n_views = str(input('\n    Enter the number of views or \'q\' to exit: '))
		
		print('*** Extracting the groups of similar clusterings. ***')
		Z       = compute_linkage_from_model(model)
		G       = np.array(cut_tree(Z, n_clusters=int(n_views)).flatten())
		
		D      = personalized_dendrogram(model, G)
	
		plot_dendrogram(D, G, resultsPath)
		
		choice    = str(input('\n    Type "ok" to proceed or press Enter to input another number: '))
		if choice in ['ok', 'OK', 'Ok', 'oK']: 
			return G
		elif choice in ['q','Q']:
			print('\nThe program in ended.')
			sys.exit()
	
	except ValueError:
		if n_views in ['q','Q']:
			print('\nThe program in ended.')
			sys.exit()
		print('\nInvalid number of views')
'''

'''
def mClustRandomProjection__(X, n_projections=60, n_clusters=2, dis_metric='dist_clusterings'):
	print('\n*** The algorithm is started with the following parameter values: \n    %d projections, each with %d clusters.\n'%(n_projections, n_clusters))
	
	P = []
	for p in range(n_projections):
		XX = copy.deepcopy(X)
		Mp = construct_projection_matrix(XX.shape[1])
		Xp = XX @ Mp
		#Xp = random_projection.GaussianRandomProjection(n_components=X.shape[1]).fit_transform(XX)
		Sp = GaussianMixture(n_components=n_clusters).fit_predict(Xp)
		P.append(Sp)
	
	P      = np.array(P)
	print('*** %d projections and clusterings are generated. ***'%(n_projections))
	
	A      = affinity(P, affinity_metric=dis_metric)
	print('*** Clusterings dissimilarity matrix is generated. ***')
	
	M      = AgglomerativeClustering( linkage="average", metric="precomputed", compute_distances=True ).fit(A)
	print('*** Clusterings hierarchy is generated with an agglomeartive model. ***') 

	return P, A, M
'''

'''
P, A, M_mdl		     = mClustRandomProjection__(
						DATA, 
						n_projections = n_projections, 
						n_clusters 	  = n_clusters, 
						dis_metric 	  = dis_metric )
D = personalized_dendrogram(M_mdl, [0 for _ in M_mdl.labels_] , resultsPath)
plot_dendrogram(D, [0 for _ in M_mdl.labels_] , resultsPath)
'''

#print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')

'''
while True:
	G = label_clusterings__(M_mdl, resultsPath)
	R = all_representatives(model= M_mdl, clusterings= P, distances= A, group_lables= G, method= method ) 

	for S_id, S in enumerate(R):
		if data_name[0:5] == 'image':
			# Coloring RGB pixels with thier cluster correspondiing color (2 colors, 1 for each cluster)
			#clr = [ [0, 0, 0], [255, 255, 255] ] 	 
			clr = [[0,0,0], [0,128,0], [255,140,0], [165,42,42], [255,255,255], [65,105,225], [255,215,0] ]
			
			
			# coloring RGB pixels with their cluster means
			#clr = [ [np.mean(col) for col in zip(*DATA[S==cl])] for cl in set(S) ] 
		else:
			# Coloring data points with thier cluster correspondiing color
			clr = ['black', 'green', 'orange', 'brown', 'white', 'cornflowerblue', 'yellow' ]
		
		sorted_labels = reorder_labels_by_cluster_size(DATA, S) 
		plot_clusters( DATA, [ clr[i] for i in sorted_labels ], S_id, resultsPath )
	
	print('*** Final solutions are presented. ***')
	
'''