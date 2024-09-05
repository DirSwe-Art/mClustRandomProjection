'''
Dirar Sweidan
Created:        2024_08_20
Last modified:  2024_  _

- This code is an implementtaion of the k-means algorithm from scratch.
'''

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy


def dist(a,b):
	s = sum([ (float(a[i]) - float(b[i]))**2 for i in range(len(a)) ])
	return (math.sqrt(s))**2

def nearestCenterIndex(x, C):
	return np.argmin( [dist(x,c) for c in C] )

def iterate(X, C):
	L = [ [] for k in range(len(C)) ] 	# list of k clusters
	Y = []                            	# list of cluster label of each x in X
	
	for x in X:							# assign each x of X to the cluster with the nearest center
		ic = nearestCenterIndex(x, C)
		L[ic].append(x)
		Y.append(ic)
	
	for i in set(Y):					# updating the centers of each cluster
		cluster = L[i]
		C[i]    = [ np.mean(col) for col in zip(*cluster) ]
	
	return C, L, Y

def kmeans(X, k=2, eps=0.0001):
	C = random.sample(X, k) 			# choose k initial cluster centers
	
	while True:
		prev_C = copy.deepcopy(C)
		C, L, Y = iterate(X, C)
		if np.mean([ dist(prev_c, c) for (prev_c, c) in zip(prev_C, C) ]) < eps:
			break
	
	return C, L, Y




'''
def kmeans(X, k=2, eps=0.0001):
	C = random.sample(X, k) # choose k initial centers
	#C = np.array(X)[np.random.randint(len(X), size=k)]
	#counter = 0
	while True:
		#counter += 1
		#print(counter, end=' ', flush=True)
'''