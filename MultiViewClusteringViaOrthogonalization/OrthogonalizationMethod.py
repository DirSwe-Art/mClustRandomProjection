"""Multi-View Clustering via Orthogonalization"""

# Authors: DirarSweidan
# License: DSB 3-Claus

import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power


# ========================================================================

def generate_data(type='4-2-2'):			# 4 features, 3 clusters, 2 views
	if type == '4-3-2': return data432()
	if type == '2-2-4': return data224()
	
def data224():
	X = []
	M = [[2, 2],[-2, 2],[-2, -2],[2, -2]]
	for m in M:
		X += np.random.multivariate_normal(m, np.identity(2)/3, size=125).tolist()
	
	X = np.array(X)
	X = X -X.mean(axis=0)					# center the data
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
	
def plot_clusters(DATA, X, colors):
	fig, ((ax1a, ax2a),(ax1b, ax2b)) = plt.subplots(2, 2)
	
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
	
	plt.show()
	
def mView_Clustering_via_Orthogonalization(DATA, alternatives):
	X   = copy.deepcopy(DATA)
	clr = ['green','yellow','black','blue']

	for t in range(alternatives):
		h = KMeans(n_clusters=k).fit(X)					# Clustering X first
		plot_clusters( DATA, X, colors= [ clr[i] for i in h.predict(X) ]) # Coloring original (DATA) and transformed (X) based on X new clustering
		if t == alternatives - 1: break
		
		for i, x in enumerate(X):						# Project X data on the space orthogonal to u vector
			u   = h.cluster_centers_[ h.predict([x])[0] ]	# Find cluster center (u) closest to x 
			uuT = np.array([u]) * np.array([u]).T
			uTu = np.dot(u, u)
			X[i]= (np.identity(len(x)) - uuT / uTu ).dot(x)
			
			'''
			u   = h.cluster_centers_[ h.predict([x])[0] ]
			u   = np.array(u)
			I   = np.odentity(len(x))
			X[i]= ( I-(u.T * u)/ u.dot(u.T) ).dot(x.T)
			'''

# ========================================================================
# Paper: Y. Cui et al. (2007). Non-redundant multi-view clustering via orthogonalization. ICDM (pp. 133-142).

#k    = 2
#DATA = generate_data(type='2-2-4')

k    = 3
DATA = generate_data(type='4-3-2')

alternatives = 5 
mView_Clustering_via_Orthogonalization(DATA, alternatives)
