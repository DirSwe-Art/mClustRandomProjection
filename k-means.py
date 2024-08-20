'''
Dirar Sweidan
Created:        2024_08_20
Last modified:  2024_  _

- This file runs the k-means algorithm from the Sk-Learn library.
'''

from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import numpy as np
import random

# ================================================================== Generate data

# Generate a dataset randomly (example with 2 clusters)
X1 = [ [np.random.normal(5,1), np.random.normal(5,1)] for u in range(500) ]
X2 = [ [np.random.normal(10,1), np.random.normal(10,1)] for u in range(500) ]

X  = X1 + X2

# ================================================================== Clustering

cl = KMeans(n_clusters=2)
cl.fit(X)

Y  = cl.predict(X) # a list of the indices of predicted clusters for each x in X
Mu = [ center for center in cl.cluster_centers_] # cluser centers: each center is a mean of cluster points 

# ================================================================== Plotting

colors = ['r','b','g','k','y']

F1, F2 = zip(*X)
m1, m2 = zip(*Mu)
plt.scatter(F1, F2, color = [colors[i] for i in Y] )
plt.scatter(m1, m2, color = [colors[i+2] for i in range(2) ] )

plt.show()
