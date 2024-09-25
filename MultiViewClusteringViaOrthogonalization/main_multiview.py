"""Multi-View Clustering via Orthogonalization"""

# Authors: DirarSweidan
# License: DSB 3-Claus

import copy, os, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power


# ========================================================================

def generate_data(type='4-3-2'):			# 4 features, 2 clusters, 2 views		
	if type == '4-3-2':
		DATA = data432()
		k = 3
		datatype = 'F-C-V'
		return DATA, k, datatype
		
	elif type == '2-2-3':
		DATA = data223()
		k = 2
		datatype = 'F-C-V'
		return DATA, k, datatype
		
	elif type == 'image':
		DATA, imRow, imCol, imDim = dataimg('source_images/image.bmp')
		k = 2
		datatype = 'image'
		return DATA, k, datatype, imRow, imCol, imDim


def data223():								# 2 features, 2 clusters, 3 views
	X = []
	M = [[2, 2],[-2, 2],[-2, -2],[2, -2]]
	for m in M:
		X += np.random.multivariate_normal(m, np.identity(2)/3, size=125).tolist()
	
	X = np.array(X)
	X = X - X.mean(axis=0)					# center the data
	return X
	
	
def data432():								# 4 features, 3 clusters, 2 views
	X1 = []									# First view of the data (with F1, F2 features of each x)
	M  = [[7, 2], [3, 7], [10, 9]]			# Three centroids
	S  = [100, 100, 300]					# The first view's number of data points in each cluster
	
	for i in range(len(S)):
		X1 += np.random.multivariate_normal(M[i], np.identity(2)/3, size=S[i]).tolist()
	
	X2 = []									# Second view of the data (with F3, F4 features of each x)
	M  = [[5, 4], [7, 11], [12, 6]]			# Three centroids
	S  = [200, 200, 100]					# The second view's number of data points in each cluster
	
	for i in range(len(S)):
		X2 += np.random.multivariate_normal(M[i], np.identity(2)/3, size=S[i]).tolist()
	
	X  = np.array([ X1[i] + X2[i] for i in range(len(X1)) ]) # Combine the two views (F1, F2, F3, F4)
	X  = X - X.mean(axis=0)					# Center the data (shift the data points toward the origine)
	return X


def dataimg(file):
	IMG = plt.imread(file)					# uint8 data type
	imRow, imCol, imDim = IMG.shape
	X = []
	
	for r in range(imRow):
		for c in range(imCol):
			X.append( IMG[r][c] )
			
	return X, imRow, imCol, imDim
	
	
def plot_clusters(DATA, X, colors, t, resultsPath):
	if datatype == 'F-C-V': return random_clusters(DATA, X, colors, t, resultsPath)
	if datatype == 'image': return image_clusters(DATA, X, colors, t, resultsPath) 
	
	
def random_clusters(DATA, X, colors, t, resultsPath):
	fig, ((ax1a, ax2a),(ax1b, ax2b)) = plt.subplots(2, 2, figsize=(15, 11), sharex=False, sharey=False)
	
	ax1a.scatter( *np.array([ *zip(*DATA) ])[:2], c=colors, marker='.' )
	ax1a.set_title('Original Space')
	ax1a.set_xlabel('Feature 1')
	ax1a.set_ylabel('Feature 2')
	
	if len(DATA[0]) > 2:
		ax2a.scatter( *np.array([ *zip(*DATA) ])[2:4], c=colors, marker='.' )
		ax2a.set_title('Original Space')
		ax2a.set_xlabel('Feature 3')
		ax2a.set_ylabel('Feature 4')
		
	ax1b.scatter( *np.array([ *zip(*X) ])[:2], c=colors, marker='.' )
	ax1b.set_title('Transformed Space')
	ax1b.set_xlabel('Feature 1')
	ax1b.set_ylabel('Feature 2')
	
	if len(DATA[0]) > 2:
		ax2b.scatter( *np.array([ *zip(*X) ])[2:4], c=colors, marker='.' )
		ax2b.set_title('Transformed Space')
		ax2b.set_xlabel('Feature 3')
		ax2b.set_ylabel('Feature 4')
	
	if len(DATA[0]) > 2:
		plt.savefig(resultsPath+'random_3_clustering_n_'+str(t+1)+'.jpg')
	else:
		plt.savefig(resultsPath+'random_2_clustering_n_'+str(t+1)+'.jpg')
	
	plt.close('all')	


def image_clusters(DATA, X, colors, t, resultsPath):
	IMG_DATA = np.array(copy.deepcopy(DATA), dtype='uint8').reshape(imRow, imCol, imDim)
	IMG_X    = np.array(copy.deepcopy(X), dtype='uint8').reshape(imRow, imCol, imDim)
	
	f, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=False, sharey=False, figsize=(6, 15))
	ax1.imshow(IMG_DATA)
	ax1.set_title('Source image')								# view the original space
	
	ax2.imshow(IMG_X)
	ax2.set_title('Transformed space')							# view the space orthogonal to the clustering solution
	
	ax3.imshow(np.array(colors).reshape(imRow, imCol, imDim)) 	# view the segmented space (center colors)
	ax3.set_title('Clustering Solution')
	
	plt.savefig(resultsPath+'image_clustering_n_'+str(t+1)+'.jpg')
	plt.close('all')	
	

# ========================================================================
# Paper: Y. Cui et al. (2007). Non-redundant multi-view clustering via orthogonalization. ICDM (pp. 133-142).
def mView_Clustering_via_Orthogonalization(DATA, alternatives, k, datatype, resultsPath):
	X   = copy.deepcopy(DATA)
	
	for t in range(alternatives):
		print('Iteration', t)
		
		h = KMeans(n_clusters=k).fit(X)												# Clustering X first
		
		if datatype == 'image': clr = np.array(h.cluster_centers_, dtype='uint8') 	# For coloring pixels of each cluster by the mean color (centroid) 
		else: 					clr = ['green','yellow','black','blue']				# For coloring data points of each cluster by a given color
		plot_clusters( DATA, X, [ clr[i] for i in h.predict(X) ], t , resultsPath)	# Coloring original (DATA) and transformed (X) based on X clustering
		
		if t == alternatives - 1: break
		
		for i, x in enumerate(X):													# Project X data on the space orthogonal to u vector
			u   = h.cluster_centers_[ h.predict([x])[0] ]							# Find cluster center (u) closest to x (u is a vector column "2×1 matrix")
			'''
			uuT = np.array([u]) * np.array([u]).T									# uuT is [2×1]*[1×2]=[2×2] matrix. The matrix multiplication
			uTu = np.dot(u, u)														# uTu is [1×2]*[2×1]=[1×1] matrix. The dot product (scalar) 
			X[i]= (np.identity(len(x)) - uuT / uTu ).dot(x)							# The projection of x on the transformation matric
			'''
			
			# another way to computing it:
			u = np.array(u).reshape(len(u),1)										# reshape u to a vector column
			uuT = np.dot(u,u.T)														# it returnes a [2×2] matirx ([2×1]*[1×2])
			uTu = np.dot(u.T,u)														# it returnes a scalar ([1×2]*[2×1]=[1×1] matrix)
			I   = np.identity(len(x))
			X[i]= (I - uuT / uTu).dot(x) 											# the transformation matrix dot product vector x (vector column) or
			#X[i]= M * np.array(x).T 												# transformation matrix mltiplied by the transpose of vector x
			
# ========================================================================

resultsPath  		= r'C:\ExperimentalResults\Results\results_MultiViewClusteringViaOrthogonalization'
if not os.path.exists(resultsPath): os.makedirs(resultsPath)


alternatives 		= 5 


#DATA, k, datatype, imRow, imCol, imDim = generate_data(type= 'image') 				# 'image'
DATA, k, datatype  	= generate_data(type= '4-3-2') 									# '2-2-3', '4-3-2'


mView_Clustering_via_Orthogonalization(DATA, alternatives, k, datatype, resultsPath)



'''
Repersenting a clustered image by foreground and background colors

list_to_uint8 = lambda A: [np.uint8(v) for v in A]
rep_colors = [ [0, 0, 0], [255, 255, 255], [128, 128, 128], [64, 64, 64], [200, 200, 200] ] + [ np.random.randint(0, 255, 3).tolist() for _ in range(100) ]

colors = [ list_to_uint8( rep_colors[i] ) for i in range(2) ]


XX = np.array([ colors[i] for i in Y ])		# projection of the original image on the new centers (float32 dtype)


### I centered image data. The first clustering was good. After the first transformation (of the centered image) the resuls was not good, and the second clustering was bad. 
### I think, centering image data is not always a good choice.
'''
