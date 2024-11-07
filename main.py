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
from scipy.sparse import lil_matrix
import pandas as pd
#import dask


# ========================================================================
def constructProjectionMatrix(d):
	# Returns a random projection transformation matrix spanned by a linearly independent (orthogonal) unit vectors. #
	
	bit_generator = np.random.PCG64DXSM()		# Create a 128-bit bit generator (PCG64DXSM)
	rng  = np.random.Generator(bit_generator)	# Create a Generator instance using the 128-bit bit generator
	M    = rng.normal(0, 1/d, size=(d,d))		# Generator's normal method to generate the matrix M
	#M    = M @ np.linalg.inv(M.T @ M) @ M.T		# Prjection matrix (projects data onto a space spanned by the unit vectors in M).#

	return M

'''
def distance(Ya, Yb):
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

def distance(Ya, Yb):
	# enhanced function to compute the actual distance between two clusterings
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

def dist_clusterings(Ya, Yb, threshold=2200):
	# Returns an approximate distance between two clusterings if the threshold
	# is not None and the data size is larger than it, otherwise it returns
	# the actual distance. 'threshold' can be adjusted according to the PC's RAM size. 
	
	if threshold == None or len(Ya) < threshold: return distance(Ya, Yb)
	
	ds_rand_Ys    = []
	for i in range(10):
		rand_ids = np.random.choice(range(len(Ya)), threshold, replace=False) # replace=False a value a is selected once.
		ds_rand_Ys.append( distance([Ya[id] for id in rand_ids], [Yb[id] for id in rand_ids]) )
	return np.mean(ds_rand_Ys)

def affinity(data, affinity_metric='dist_clusterings'):
	if   affinity_metric == 'dist_clusterings':             return pairwise_distances(data, metric=dist_clusterings)
	# We can add more metrics
	# elif affinity_metric == 'hamming_dist': return ...  #

def central(clusterings, distances):
	# returns a clustering from the pool that has the minimum sum of distnaces with all other clutserings. #
	A      = distances
	id_min = np.argmin([ sum(A[row_id]) for row_id in range(len(A)) ])
	
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
	# This is the original (lazy) clustering aggragation function. It suffurs 
	# from the lack of memory with extreame dimensions data, e.g., 1000000-D. 
	
	# It returns a clustering solution where the label of each data point is
	# estimated from a matrix representation of the dataset. Each element (i,j)
	# represents the number of clusterings where x_i, x_j are grouped together.

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

'''

def aggregate(clusterings):
	# This shows a better performance with a larger RAM. Still, it suffurs 
	# from the lack of memory with extreame dimensions data, e.g., 1000000-D. 
	
	# It returns a clustering solution where the label of each data point is
	# estimated from a matrix representation of the dataset. Each element (i,j)
	# represents the number of clusterings where x_i, x_j are grouped together.
	
	# dictionary for sample pairwise equality comparisons in each clustering
	dict_ = {}
	for s_id, S in enumerate(clusterings):
		dict_[s_id]= np.equal.outer(S,S)

	xS = [] # xi and each xj are together (m-element row for each solution)
	xC = [] # xi representation (the sum of xj-column over all solutions where xi,xj are together)

	for x_id in range(len(clusterings[0])):
		for s_id, S in enumerate(clusterings):
			xS.append(dict_[s_id][x_id])
		sums  = np.sum(np.array(xS), axis=0)
		xC.append(sums)
		xS    = []
		
	return GaussianMixture(n_components=len(set(clusterings[0]))).fit_predict(xC).tolist()

def aggregate_large_data(clusterings):
	# Returns an aggregated representative solution for all solutions in clusterings.
	# This function solves the memory lack issue with large datasets. 
	# However, it is effective but not efficient, meaning that it takes a long
	# processing time since operations are not accomplished directly through 
	# the RAM, but through a virtual memory on the hard disk.
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

	def allOccurance(clusterings, result_path):
		# Returns one dictionary saved externally for all solutions in clusterings.  
		# Each key is for one solution's elements pairwise equality comparison.
		with h5py.File(result_path, 'w') as hf:
			for s_id, S in enumerate(clusterings):
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
				os.remove('S'+str(s_id)+'_1')
				os.remove('S'+str(s_id)+'_2')
				os.remove(SS_result_path)
				time.sleep(1)
				
				# After this loop, the entire memmap data will be written to the HDF5 file
			
				print('\t -> Pairwise occurancies in solution %d'%s_id)
		return result_path

	def occuranceFreq(x_id, clusterings, clusterings_dict_path):
		# Returns a vector representation of one data point x where frequencies  
		# of which it occures together with other points across solutions in clusterings.
		with h5py.File(clusterings_dict_path, 'r') as hf:
			xS = pd.DataFrame( {}, columns=range(len(clusterings)), dtype=np.int8)
			for s_id, S in enumerate(clusterings):
				xS[str(s_id)] = hf['S'+str(s_id)][x_id]
			return xS.sum(axis=1)

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

	
	clusterings_dict_path = allOccurance(clusterings, 'clusterings_dict') # => clusterings_dict.h5 
	xC          = np.memmap('clusterings_matrix', dtype=np.int8, mode='w+', shape=(len(clusterings[0]),len(clusterings[0])))
	
	print('\n\t -> Occurance frequencies X^C (%d, %d) for all X(xi, xj) in all clusterings(S). Save in an external file.'%(len(clusterings[0]),len(clusterings[0])) )
	for x_id in range(len(clusterings[0])):
		freq 		= occuranceFreq(x_id, clusterings, clusterings_dict_path)
		xC[x_id, :]	= freq
		if x_id % 10000 == 0: print('\t -> batch', x_id)
	
	xC.flush(); del xC ; time.sleep(1)
														
	print('\n\t -> Emptying the RAM, reading the external X^C file')
	
	xC_memory   = np.memmap('clusterings_matrix', dtype=np.int8, mode='r', shape=(len(clusterings[0]),len(clusterings[0])))
	kmeans      = MiniBatchKMeans(n_clusters=len(set(clusterings[0])), batch_size=10000, max_iter=100, tol=1e-4, max_no_improvement=15, random_state=42)
	kmeans      = batch_fit_kmeans(kmeans, xC_memory, batch_size=10000)
	predictions = batch_predict(   kmeans, xC_memory, batch_size=10000)
	del xC_memory
	os.remove(clusterings_dict_path)
	os.remove('clusterings_matrix')
	
	return predictions

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
	plt.xlabel('Sample label')
	plt.ylabel('Distance')
	plt.savefig(resultsPath+'dendrogram_'+data_name[6:]+'_k_'+str(n_clusters)+'.jpg')
	plt.show()

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

	return P, A, M

def get_groups_of_solutions(model):
	while True:
		try:
			n_views = str(input('\n    Enter the number of views or \'q\' to exit: '))
			
			print('*** Extracting the groups of similar clusterings. ***')
			Z       = computeLinkageFromModel(model)
			G       = np.array(cut_tree(Z, n_clusters=int(n_views)).flatten())
			
			plotDendrogram(model, G, resultsPath)
			
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

def representative_solutions(model, clusterings, distances, group_lables, method='aggregate'):
	print('*** Aggregating groups of clusterings is started. ***')
	
	R      = []
	for label in set(group_lables):
		ids= np.asarray(group_lables==label).nonzero()
		C  = clusterings[ids]
		AC = distances[ids,ids]
		
		if len(C) == 1:
			print('*** Group (%d) has one clustering solution. No aggregation is needed. ***'%label)
			R.append(C[0].tolist())
		elif method == 'central': 
			print('*** Finding the central solution of group (%d). ***'%label)
			R.append(central(C, AC))
		elif method == 'ensemble': 
			print('*** Computing the ensemble solution of group (%d). ***'%label)
			R.append(ensemble(C))
		elif method == 'aggregate': 
			print('*** Aggregating clusterings of group (%d). ***'%label)
			R.append(aggregate(C))
		elif method == 'aggregate_large_data': 
			print('*** Aggregating clusterings of group (%d). ***'%label)
			R.append(aggregate_large_data(C))

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

## Settings
starting_time        = time.time()
resultsPath          = r'./ExperimentalResults/'

n_projections 		 = 60
n_clusters           = 2
n_views              = 3
dis_metric			 = 'dist_clusterings'		            # 'distance', 'dist_clusterings'
rep_method 	 		 = 'aggregate'				            # 'central', 'ensemble', 'aggregate', 'aggregate_large_data'

if not os.path.exists(resultsPath): os.makedirs(resultsPath)

## Generate Data
(DATA, 
 n_clusters, 
 data_name,   
# imRow, imCol, imDim)= generate_data(data_name= 'image_000.bmp', format='bmp')	# 'image1.png', 'image2.png', 'image3.png', 'image4.png', 'image-x-ray-chest.bmp', 'image_chest_new.bmp'
 					)= generate_data(data_name= 'random223')	                # 'random432', 'random223'


P, A, M_mdl		     = mClustRandomProjection(
						DATA, 
						n_projections = n_projections, 
						n_clusters 	  = n_clusters, 
						dis_metric 	  = dis_metric )
plotDendrogram(M_mdl, [0 for _ in M_mdl.labels_] , resultsPath)

#print('\n*** duration',datetime.timedelta(seconds=(time.time()-starting_time)),' ***')

while True:
	G = get_groups_of_solutions(M_mdl)
	R = representative_solutions(model= M_mdl, clusterings= P, distances= A, group_lables= G, method= rep_method ) 

	for S_id, S in enumerate(R):
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