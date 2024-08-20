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


# ================================================================== Generate data

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
