'''
Dirar Sweidan
Created:        2024_08_20
Last modified:  2024_  _

- This file runs the k-means algorithm from the Sk-Learn library.
'''

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy


def dist(a,b):
	s = sum([ (a[i] - b[i])**2 for i in range(len(a)) ])
	return (math.sqrt(s))**2

def nearestCenterIndex(x, C):
	return np.argmin( [dist(x,c) for c in C] )

def iterate(X, C):
	L = [ [] for j in range(len(C)) ] # list of k empty lists:
	Y = []                            # empty list of cluster label of each x in X
	
	for x in X:
		ic = nearestCenterIndex(x, C)
		L[ic].append(x)
		Y.append(ic)
	
	# updating the centers:
	for i in range(len(C)):
		cluster = L[i]
		C[i] = [ np.mean(col) for col in zip(*cluster) ]
	
	return C, L, Y

def kmeans(X, k=2, eps=0.0001):
	C = random.sample(X, k) # choose k initial centers
	
	counter = 0
	while True:
		counter += 1
		print(counter)
		
		prev_C = C.copy()
		C, L, Y = iterate(X, C)
		if np.mean([ dist(prev_c, c) for (prev_c, c) in zip(prev_C, C) ]) < eps:
			break
	
	return C, L, Y

