def hierarchical(DATA, n_clusters=2, linkage='average', affinity='euclidean'):
	# returns to a clustering (model); model.clusterings, model.linkageMatrix, model.labels, model.centers
	####################### help libraries
	from itertools import combinations
	import numpy as np
	import math
	import copy
	from scipy.cluster.hierarchy import dendrogram
	####################### help functions
	def euc(a,b):
		return math.sqrt( sum( [ (a[i]-b[i])**2 for i in range(len(a)) ] ) )

	def man(a,b):
		return sum( [ abs(a[i]-b[i]) for i in range(len(a)) ] )
		
	def dist(a,b):
		if affinity =='euclidean': return euc(a,b)
		elif affinity =='manhatten': return man(a,b)
		
	def findClosestClusters(clusters, method, metric):				# clusters argument is a list of dictionaries{id, elements, ids, used}		
		S = []
		comb_ids = list( combinations( range(len(clusters)), 2 ) )

		if method == 'centroid':
			for i1, i2 in comb_ids:
				mean1= [ np.mean(col) for col in zip(*clusters[i1]['elements']) ]
				mean2= [ np.mean(col) for col in zip(*clusters[i2]['elements']) ]
				S.append([dist(mean1,mean2), i1, i2])	# clusters mean distance, clusters id
			minID = np.argmin(np.array(S)[:,0])
			return S[minID][0], S[minID][1], S[minID][2]
			
		else:
			for i1, i2 in comb_ids:
				m, n = len(clusters[i1]['elements']), len(clusters[i2]['elements'])
				
				if m == 1 and n == 1:					# if clusters contains one element
					S.append([dist(clusters[i1]['elements'][0], clusters[i2]['elements'][0]), i1, i2])
				else:
					S2 = np.zeros((m,n))
					for i in range(m):
						for j in range(n):
							S2[i][j] = dist( (clusters[i1]['elements'])[i], (clusters[i2]['elements'])[j] )
					
					if method   == 'single': S.append([np.amin(S2), i1, i2])		# minimum distance, clusters id
					elif method == 'complete': S.append([np.amax(S2), i1, i2])		# maximum distance, clusters id
					elif method == 'average': S.append([np.mean(S2), i1, i2])		# average distance, clusters id
					
			minID = np.argmin(np.array(S)[:,0])			# closest clusters
			return S[minID][0], S[minID][1], S[minID][2]

	def labels(X, clusters_i):	#arguments are: (dataset) and (clusters of datapoints ids)
		labels = []
		for i in range(len(X)):
			for k, cluster in enumerate(clusters_i):
				for idx in cluster:
					if idx == i: labels.append(k); break
				if len(labels) == i+1: break
		print('\n==> Lables are computed')
		return labels

	def centers(clusters):		#clusters of datapoints
		C = [ [] for cluster in clusters ]
		for i, cluster in enumerate(clusters):
			C[i] = [ np.mean(col) for col in zip(*cluster) ]
		print('==> Centroids are computed')
		return C
		
	####################### program	
	linkageMatrix =[]
	clusterings   =[]
	outputLabels  =[]
	
	X = copy.deepcopy(DATA); print('==> Hierarchical clustering is started... \n    Linkage method is:',linkage,'\n    Affinity is:',affinity, '\n')
	
	
	# pool clusters (starts with initial clusters)
	clusters = [ {'id':i, 'ids':[i], 'elements': [x], 'use':1} for i, x in enumerate(X) ]
	
	# dictionary of first clustering
	clusterings.append({'n_level':		0,
						'n_clusts':		len(clusters),
						'clusters_x':	[ cl['elements'] for cl in clusters if cl['use']==1 ],		# clusters of datapoints
						'clusters_i':	[ cl['ids'] for cl in clusters if cl['use']==1 ]			# clusters of ids
						})
	
	
	n_clusts = len(clusters)																		# number of clusters in the pool
	counter=0
	while n_clusts > 1:		
		clustList = [ cl for cl in clusters if cl['use']==1 ]										# clusters in the pool that can be used
		dis,c1,c2 = findClosestClusters(clustList, linkage, affinity)								# find closest two clusters with distance
		
		# combine two clusters in one (new)
		newClust  = {'id':len(clusters), 															# new cluster id (in the pool)															
					 'ids': clustList[c1]['ids'] + clustList[c2]['ids'], 							# combine closet clusters ids in one
					 'elements': clustList[c1]['elements'] + clustList[c2]['elements'], 			# combine closest clusters elements in one
					 'use':1}																		# flag it as "can be used"

		# append the new cluster to the pool
		clusters.append(newClust)
		
		# flag the just combined two clusters as "can not be used"
		clusters[ clustList[c1]['id'] ]['use'] = 0; clusters[ clustList[c2]['id'] ]['use'] = 0
		
		# update the lenght of the pool 
		n_clusts = len([ x for x in clusters if x['use'] == 1 ])
		counter+=1
		
		# construct the linkage matrix (1st cluster, 2nd cluster, distance, number of observations)
		linkageMatrix.append([ clusters[ clustList[c1]['id'] ]['id'], clusters[ clustList[c2]['id'] ]['id'], dis, len(newClust['elements']) ])
		
		# append the updated pool as a new dictionary to the list of clustersings 
		clusterings.append({'n_level': 	 	int(counter), 		# storing resulted clustering as a new dictionary
							'n_clusts':		n_clusts,
							'clusters_x':	[ x['elements'] for x in clusters if x['use'] == 1 ],
							'clusters_i':	[ x['ids'] for x in clusters if x['use'] == 1 ]})
		
		# find the required number of clusters and the corresponding labels and centers
		if n_clusts == n_clusters:
			outputLabels = labels(DATA, clusterings[-1]['clusters_i'] )
			outputCenters = centers(clusterings[-1]['clusters_x'])

	print('==> Clustering is completed \n    Total clusterings are ', len(clusterings), '\n    Linkage matrix is ready.')
	
	# construct a class to save the clustering model
	class model:
		def __init__(self):
			self.clusterings 	= clusterings
			self.linkageMatrix 	= np.array(linkageMatrix)
			self.labels			= outputLabels
			self.centers 		= outputCenters
		
	return model()

def plotDendogram(Z, **kwargs):
	from scipy.cluster.hierarchy import dendrogram
	import numpy as np
	linkage_matrix = Z							# columns 0,1: children, 2: distances, 3: number of observations
	linkage_matrix[:,2] = np.arange(Z.shape[0])	# make the distance rate is 1

	print('    Plotting ... \n')
	plt.figure(figsize=(10, 6))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dendrogram(	linkage_matrix, 				# Z if we use the original distances
				leaf_rotation=90,
				leaf_font_size=8,
				#show_leaf_counts=True,
				#truncate_mode='lastp', # show only the last p merged clusters
				#p=12,  
				#show_contracted=True,	# to get a distribution impression in truncated branches
				**kwargs
				)
	plt.show()




##################################################### Run the program


from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

DATA = datasets.make_moons(n_samples=150, shuffle=True, noise=0.05 , random_state=None)[0]
plt.scatter(DATA[:,0], DATA[:,1])
plt.show()

clr = ['r', 'b', 'k']

model = hierarchical(DATA, n_clusters=2, linkage='single', affinity='euclidean')

Z = model.linkageMatrix
Y = model.labels

plotDendogram(Z, labels=Y)

plt.scatter(DATA[:,0], DATA[:,1], c = [ clr[y] for y in Y ])
plt.show()

##################################################### Data

def prepare_image(image):
	import matplotlib.image as mpimg
	from matplotlib import pyplot as plt
	
	IMG = mpimg.imread(image)		# uint8 data type
	imRow, imCol, imDim = IMG.shape
	DATA = []

	for r in range(imRow):
		for c in range(imCol):
			DATA.append( IMG[r][c] )
	
	DATA = np.array(DATA, dtype='float32')
	print('\n==> Data is prepared \n')
	return DATA, imRow, imCol, imDim

def display_Image(DATA, output_image, imRow, imCol, imDim):
	import matplotlib.image as mpimg
	from matplotlib import pyplot as plt
	
	DATA = np.array(DATA, dtype='uint8')
	output_image = np.array(output_image, dtype='uint8')
	IMG_DATA = DATA.reshape(imRow, imCol, imDim)
	IMG_output = output_image.reshape(imRow, imCol, imDim)
		
	f, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True)
	ax1.imshow(IMG_DATA); ax1.set_title('Source image')	# view the original space
	ax2.imshow(IMG_output); ax2.set_title('clusters')	# view the clusters space
	plt.xticks([]); plt.yticks([])
	#plt.savefig('a= '+str(a)+'image_'+str(t+1)+'.jpg')
	print('==> Image is displayed \n')
	plt.show()

#####################################################