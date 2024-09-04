"""A principled and flexible framework for finding alternative clusterings"""

# Authors: DirarSweidan
# License: DSB 3-Claus

import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power


# ========================================================================

def generate_data(type='4-2-2'):			# 4 features, 2 clusters, 2 views
	import os
	if not os.path.exists('results'): os.makedirs('results')
	
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
		DATA, imRow, imCol, imDim = dataimg('source_images/image_.bmp')
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
	S  = [100, 100, 300]					# The number of data points in each custer in the first view
	
	for i in range(len(S)):
		X1 += np.random.multivariate_normal(M[i], np.identity(2)/3, size=S[i]).tolist()
	
	X2 = []									# Second view of the data (with F3, F4 features of each x)
	M  = [[5, 4], [7, 11], [12, 6]]			# Three centroids
	S  = [200, 200, 100]
	
	for i in range(len(S)):
		X2 += np.random.multivariate_normal(M[i], np.identity(2)/3, size=S[i]).tolist()
	
	X  = np.array([ X1[i] + X2[i] for i in range(len(X1)) ]) # Combine the two views (four features F1, F2, F3, F4)
	X  = X - X.mean(axis=0)					# Center the data (shift the data points toward the origine)
	return X


def dataimg(file):
	IMG = plt.imread(file)					# uint8 data type
	imRow, imCol, imDim = IMG.shape
	X = []
	
	for r in range(imRow):
		for c in range(imCol):
			X.append( IMG[r][c] )
	
	X = np.array(X, dtype='uint8')	
	return X, imRow, imCol, imDim
	
	
def plot_clusters(DATA, X, colors, t):
	if datatype == 'F-C-V': return random_clusters(DATA, X, colors, t)
	if datatype == 'image': return image_clusters(DATA, X, colors, t) 
	
	
def random_clusters(DATA, X, colors, t):
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
		plt.savefig(r'results/random_3_clustering_n_'+str(t+1)+'.jpg')
	else:
		plt.savefig(r'results/random_2_clustering_n_'+str(t+1)+'.jpg')
	
	plt.close('all')	


def image_clusters(DATA, X, colors, t):
	IMG_DATA = np.array(copy.deepcopy(DATA), dtype='uint8').reshape(imRow, imCol, imDim)
	IMG_X    = np.array(copy.deepcopy(X), dtype='uint8').reshape(imRow, imCol, imDim)
	
	f, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=False, sharey=False, figsize=(6, 15))
	ax1.imshow(IMG_DATA)
	ax1.set_title('Source image')								# view the original space
	
	ax2.imshow(IMG_X)
	ax2.set_title('Transformed space')							# view the transformed space (to the space orthogonal to the clustering solution)
	
	ax3.imshow(np.array(colors).reshape(imRow, imCol, imDim))	# view the segmented space (center colors)
	ax3.set_title('Clustering Solution')
	
	plt.savefig(r'results/image_clustering_n_'+str(t+1)+'.jpg')
	plt.close('all')	
	

# ================================================================================
# Paper: Z. Qi et al. (2009). A principled and flexible framework for finding alternative clusterings. SIGKDD (pp. 717-726).
def princ_flex_framework_AlternativeClustering(DATA, a, alternatives, k, datatype):
	X = copy.deepcopy(DATA)
	
	for t in range(alternatives):
		print('Iteration', t)
		
		h = KMeans(n_clusters= k).fit(X)
		
		if datatype == 'image': clr = np.array(h.cluster_centers_, dtype='uint8') 	# For coloring pixels of each cluster by the mean color (centroid) 
		else: 					clr = ['green','yellow','black','blue']				# For coloring data points of each cluster by a given color
		plot_clusters( DATA, X, [ clr[i] for i in h.predict(X) ], t )				# Coloring original (DATA) and transformed (X) based on X clustering
		
		S = np.zeros( (len(DATA[0]),len(DATA[0])) ) # Sigma: data point variation for (k-1) centroids where this data point unlikely belongs to PI'
		
		# B= D*D' and B=(Sigma)^(-1) => D=sqrt(B)=(Sigma)^(-1/2)
		for i, x in enumerate(X):
			# List of centroieds that x does not belong to
			C = [ u for iu, u in enumerate(h.cluster_centers_) if iu != h.predict([x])[0]] 
			
			# Add the distances between xi and all those centroids to the Sigma matrix 
			for u in C:
				S += (x-u) * np.array([(x-u)]).T
		
		# divide by the number of data points: 
		# Sigma= 1/n(xi-mj)*(xi-mj)' where xi does not belong to Cj
		S = S / len(X)
		
		# Transformation matrix: B=(Sigma)^(-1), a=2, B=(Sigma)^(-a/2), D=sqrt(B)=(Sigma)^(-a/4)
		D = fractional_matrix_power(S, -a/4.)	
		
		# Transforming X to Y by D: Y=D.X
		X = (D.dot(X.T)).T
		
# ================================================================================

alternatives = 5
a = 2

#DATA, k, datatype  = generate_data(type= '2-2-4') 						# '2-2-4', '4-3-2'
DATA, k, datatype, imRow, imCol, imDim = generate_data(type= 'image') 	# 'image'

princ_flex_framework_AlternativeClustering(DATA, a, alternatives, k, datatype)


