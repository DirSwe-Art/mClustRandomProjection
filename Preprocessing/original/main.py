from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pylab as plt
import numpy as np
from utils import plot_data, plot_dendrogram, show_image, dataset_artificial, dataset_image
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import random_projection

# ================================================================================
def dist_clusterings( Ya, Yb ):
	if len(Ya) > 1000:
		ids = np.random.choice(range(len(Ya)), 1000, replace=False)
		Ya = [Ya[i] for i in ids]
		Yb = [Yb[i] for i in ids]
	
	d = 0
	comb_ids = list( combinations( range(len(Ya)), 2 ) )
	for i1, i2 in comb_ids:
		if (Ya[i1] == Ya[i2] and Yb[i1] != Yb[i2]) or (Yb[i1] == Yb[i2] and Ya[i1] != Ya[i2]):
			d += 1
	return d
	
# ================================================================================
def affinity( data ):
	return pairwise_distances( data, metric = dist_clusterings )

# ================================================================================
# X = dataset_artificial()
X, image_shape = dataset_image("img1.png")

clusterings = [] # the list of clustering results
for i in range(10):
	print("clustering", i)
	
	XX = random_projection.GaussianRandomProjection(n_components = X.shape[1]).fit_transform(X)
	
	clu = GaussianMixture(n_components=2).fit(XX)
	Y = clu.predict(XX)
	show_image( clu.means_, Y, image_shape )
	
	clusterings.append( Y )
	
	# plot_data(X, Y)
	# plot_data(XX, Y)
#

# ================================================================================
model = AgglomerativeClustering( n_clusters=2, linkage="average", affinity=affinity ).fit( clusterings )
Y = model.labels_
print("Predicted clusters for the clustering results", Y)

plot_dendrogram(model, labels=Y)
plt.show()



