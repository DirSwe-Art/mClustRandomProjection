'''
Dirar Sweidan
Created:        2024_08_20
Last modified:  2024_  _

- This file containes clustering examples of random data.
'''

import matplotlib.pyplot as plt
import numpy as np
import random
from kMeansFromScratch import kmeans
from sklearn.cluster import KMeans


# ================================================================== Generate data

import os
if not os.path.exists('results'): os.makedirs('results')
	
# Generate a dataset randomly (example with 2 clusters)
X1 = [ [np.random.normal(5,1), np.random.normal(5,1)] for u in range(500) ]
X2 = [ [np.random.normal(10,1), np.random.normal(10,1)] for u in range(500) ]

X  = X1 + X2
random.shuffle(X)

F1, F2 = zip(*X)
plt.scatter(F1, F2, color='g')
plt.show()
plt.close('all')


# ================================================================== Clustering X

C, L, Y = kmeans(X, k=2, eps=0.0001)
print('Final centers are:', C)


colors = ['r','b','g','k','y']
# plotting using the labels
F1, F2 = zip(*X)
m1, m2 = zip(*C)

plt.scatter(F1, F2, color= [ colors[i] for i in Y])
plt.scatter(m1, m2, color= [ colors[i+2] for i in range(2)])
plt.show()
plt.close('all')

# plotting using the actual clusters
for i, cluster in enumerate(L):
	F1, F2 = zip(*cluster)
	plt.scatter( F1, F2, color=colors[i] )
	
	F1m = C[i][0]
	F2m = C[i][1]
	plt.scatter(F1m, F2m, color=colors[i+2], marker='<' )

plt.show()


# ================================================================== Clustering X

# using the sk-learn library

cl = KMeans(n_clusters=2)
cl.fit(X)

Y = cl.predict(X)										# a list of the indices of predicted cluster for each x in X
centers = [ center for center in cl.cluster_centers_ ] 	# cluster centers: each center is a mean of cluster points


colors = ["r", "b", "g", "k", "y"]

F1, F2 = zip(*X)
m1, m2 = zip(*centers)
plt.scatter( F1, F2, color = [colors[i] for i in Y] )
plt.scatter( m1, m2, color = [colors[i+3] for i in range(2)] )

plt.show()
