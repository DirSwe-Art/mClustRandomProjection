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

# ================================================================== Generate data

# Generate a dataset randomly (example with 2 clusters)
X1 = [ [np.random.normal(5,1), np.random.normal(5,1)] for u in range(500) ]
X2 = [ [np.random.normal(10,1), np.random.normal(10,1)] for u in range(500) ]

X  = X1 + X2
random.shuffle(X)

F1, F2 = zip(*X)
plt.scatter(F1, F2, color='g')
plt.show(block=True)

# ================================================================== Functions

def dist(a,b):
	s = sum([ (a[i] - b[i])**2 for i in range(len(a)) ])
	return (math.sqrt(s))**2

def nearestCenterIndex(x, C):
	return np.argmin( [dist(x,c) for c in C] )

def iterate(X, C):
	L = [ [] for j in range(len(C)) ] # list of k empty lists:
	
	for x in X:
		ic = nearestCenterIndex(x, C)
		L[ic].append(x)
	
	# updating the centers:
	for i in range(len(C)):
		cluster = L[i]
		C[i] = [ np.mean(col) for col in zip(*cluster) ]
	
	return C, L

def kmeans(X, k=2, eps=0.0001):
	C = random.sample(X, k) # choose k initial centers
	
	counter = 0
	while True:
		counter += 1
		print(counter)
		
		prev_C = C.copy()
		C, L = iterate(X, C)
		if np.mean([ dist(prev_c, c) for (prev_c, c) in zip(prev_C, C) ]) < eps:
			break
	
	return C, L


# ================================================================== Clustering X

C, L = kmeans(X, k=2, eps=0.0001)
print('Final centers are:', C)

colors = ['r','b','g','k','y']
for i, cluster in enumerate(L):
	F1, F2 = zip(*cluster)
	plt.scatter( F1, F2, color=colors[i] )
	
	F1m = C[i][0]
	F2m = C[i][1]
	plt.scatter(F1m, F2m, color=colors[i+2], marker='<' )

plt.show()
