from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from itertools import combinations
import numpy as np
import math
import copy
import random
import csv
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram

##################################################### Data
def prepareData():
	X0 = [ [1,1], [1.2,1], [1.1,1.3], [1.6,1.8], [2.1,3], [1.9,3.1], [1.1,1.1], [1.05,1.15], [1.95,2.95] ]
	
	X1 = [ [np.random.normal(1.5, 0.1), np.random.normal(1, 0.1)] for u in range(35) ]
	X2 = [ [np.random.normal(1.5, 0.15), np.random.normal(-1.5, 0.15)] for u in range(35) ]
	X3 = [ [np.random.normal(-1.5, 0.05), np.random.normal(1, 0.05)] for u in range(35) ]
	
	X5 = list(datasets.make_circles(n_samples=200, shuffle=True, noise=0.05 , random_state=None, factor=0.5))
	X5 = X5[0]
	
	XX = np.concatenate((X1,X2,X3),axis=0)		#; random.shuffle(X5)
	X6 = datasets.make_moons(n_samples=300, shuffle=True, noise=0.05 , random_state=None)
	X6 = X6[0]
	
	print('\n==> Data is prepared \n')
	return np.array(X6)

def prepare_image(image):
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
	
def euc(a,b):
	return math.sqrt( sum( [ (a[i]-b[i])**2 for i in range(len(a)) ] ) )

def man(a,b):
	return sum( [ abs(a[i]-b[i]) for i in range(len(a)) ] )
	
def findClosestClusters(clusters, method, metric):				# clusters argument is a list of dictionaries{id, elements, ids, used}
	def dist(a,b):
		if metric =='euclidean': return euc(a,b)
		elif metric =='manhatten': return man(a,b)
		
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

def hierarchical(DATA, n_clusters=2, linkage='average', affinity='euclidean'):
	linkageMatrix =[]
	clusterings   =[]
	outputLabels  =[]
	
	X = copy.deepcopy(DATA); print('==> Hierarchical clustering is started... \n    Linkage method is:',linkage,'\n    Affinity is:',affinity, '\n')
	clusters = [ {'id':i, 'ids':[i], 'elements': [x], 'use':1} for i, x in enumerate(X) ]
	clusterings.append({'n_level':		0, 		# dictionary of first clustering (each data point is a cluster)
						'n_clusts':		len(clusters),
						'clusters_x':	[ cl['elements'] for cl in clusters if cl['use']==1 ],
						'clusters_i':	[ cl['ids'] for cl in clusters if cl['use']==1 ]
						})
	n_clusts = len(clusters)
	counter=0
	while n_clusts > 1:		
		clustList = [ cl for cl in clusters if cl['use']==1 ]
		dis,c1,c2 = findClosestClusters(clustList, linkage, affinity)
		newClust  = {'id':len(clusters), 
					 'ids': clustList[c1]['ids'] + clustList[c2]['ids'], 
					 'elements': clustList[c1]['elements'] + clustList[c2]['elements'], 
					 'use':1}

		clusters.append(newClust); clusters[ clustList[c1]['id'] ]['use'] = 0; clusters[ clustList[c2]['id'] ]['use'] = 0
		n_clusts = len([ x for x in clusters if x['use'] == 1 ])
		counter+=1
		linkageMatrix.append([ clusters[ clustList[c1]['id'] ]['id'], clusters[ clustList[c2]['id'] ]['id'], dis, len(newClust['elements']) ])
		
		
		clusterings.append({'n_level': 	 	int(counter), 		# storing resulted clustering as a new dictionary
							'n_clusts':		n_clusts,
							'clusters_x':	[ x['elements'] for x in clusters if x['use'] == 1 ],
							'clusters_i':	[ x['ids'] for x in clusters if x['use'] == 1 ]})
		
		if n_clusts == n_clusters:
			outputLabels = labels(DATA, clusterings[-1]['clusters_i'] )
			outputCenters = centers(clusterings[-1]['clusters_x'])

	print('==> Clustering is completed \n    Total clusterings are ', len(clusterings), '\n    Linkage matrix is ready.')
	
	class model:
		def __init__(self):
			self.clusterings_ 	= clusterings
			self.linkageMatrix_ = np.array(linkageMatrix)
			self.labels_ 		= outputLabels
			self.centers_ 		= outputCenters
		
	return model()
	#return clusterings, np.array(linkageMatrix), outputLabels, outputCenters
		
def labels(X, clusters_i):
	labels = []
	for i in range(len(X)):
		for k, cluster in enumerate(clusters_i):
			for idx in cluster:
				if idx == i: labels.append(k); break
			if len(labels) == i+1: break
	print('\n==> Lables are computed')
	return labels

def centers(clusters):
	C = [ [] for cluster in clusters ]
	for i, cluster in enumerate(clusters):
		C[i] = [ np.mean(col) for col in zip(*cluster) ]
	print('==> Centroids are computed')
	return C

def plotDendogram(Z, **kwargs):
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
print("\nProgram is started")


#Artificial data
DATA = prepareData()
plt.scatter(DATA[:,0], DATA[:,1])
plt.show()

'''
#Image data
image = "img2.bmp"
DATA, imRow, imCol, imDim = prepare_image(image)
'''
model = hierarchical(DATA, n_clusters=2, linkage='single', affinity='euclidean')
clusteringsDictionary = model.clusterings_
Z = model.linkageMatrix_
Y = model.labels_

#clusteringsDictionary, Z, Y = hierarchical(DATA, n_clusters=2, linkage='single', affinity='euclidean')
#plotDendogram(Z, labels=Y)

while True:
	try:
		numberOfClusters = str(input('    Enter the number of clusters you want to view:'))
		dictionary = [ dict for dict in clusteringsDictionary if dict['n_clusts'] == int(numberOfClusters)]
		
		
		clust_x = dictionary[0]['clusters_x']
		clust_i = dictionary[0]['clusters_i']
		
		Y = labels(DATA,clust_i)
		
		
		for i, clust in enumerate(clust_x):
			F1, F2 = zip(*clust)
			plt.scatter( F1, F2)
		'''
		
		C = centers(clust_x)
		clusteredImage = np.array([ C[i] for i in Y ])
		display_Image(DATA, clusteredImage, imRow, imCol, imDim)
		'''
		plotDendogram(Z, labels=Y)
		plt.show()
	except ValueError:
		if numberOfClusters == 'q': print("\nProgram is ended"); break
		print("Invalid number of clusters")