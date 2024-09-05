from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from itertools import combinations
import numpy as np
import math
import copy
import random
import csv
from sklearn import datasets

##################################################### Data
def prepareData():
	X1 = [ [np.random.normal(1.5, 0.1), np.random.normal(1, 0.1)] for u in range(35) ]
	X2 = [ [np.random.normal(1.5, 0.15), np.random.normal(-1.5, 0.15)] for u in range(35) ]
	X3 = [ [np.random.normal(-1.5, 0.05), np.random.normal(1, 0.05)] for u in range(35) ]
	
	X5 = np.concatenate((X1,X2),axis=0); random.shuffle(X5)
	
	X6 = list(datasets.make_circles(n_samples=200, shuffle=True, noise=0.05 , random_state=None, factor=0.5))
	X6 = np.array(X6[0])

	XX = np.concatenate((X5,X6),axis=0)
	X7 = [ [1,1], [1.2,1], [1.1,1.3], [1.6,1.8], [2.1,3], [1.9,3.1], [1.1,1.1], [1.05,1.15], [1.95,2.95] ]

	X = []
	M = [[2, 2],[-2, 2],[-2, -2],[2, -2]]
	for m in M:
		X += np.random.multivariate_normal(m, np.identity(2)/3, size=125).tolist()
	
	X = np.array(X)
	X = X - X.mean(axis=0)					# center the data
	
	return np.array(X)

def prepare_image(image):
	IMG = mpimg.imread(image)		# uint8 data type
	imRow, imCol, imDim = IMG.shape
	DATA = []

	for r in range(imRow):
		for c in range(imCol):
			DATA.append( IMG[r][c] )
	
	DATA = np.array(DATA, dtype='float32')
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
	plt.show()

#####################################################
	
def dist(a,b):
	return math.sqrt( sum( [ (a[i]-b[i])**2 for i in range(len(a)) ] ) )
	
def singleLinkage(clusters):
	S = []
	comb_ids = list( combinations( range(len(clusters)), 2 ) )
	for i1, i2 in comb_ids:
		m, n = len(clusters[i1]), len(clusters[i2])
		
		if m == 1 and n == 1:
			S.append([dist(clusters[i1][0], clusters[i2][0]), i1, i2])
		else:
			S2 = np.zeros((m,n))
			for i in range(m):
				for j in range(n):
					S2[i][j] = dist( (clusters[i1][i]), clusters[i2][j] )
			
			S.append([np.amin(S2), i1, i2])		# minimum distance, clusters id
	minID = np.argmin(np.array(S)[:,0])			# closest clusters
	return S[minID][0], S[minID][1], S[minID][2]

def completeLinkage(clusters):
	S = []
	comb_ids = list( combinations( range(len(clusters)), 2 ) )
	for i1, i2 in comb_ids:
		m, n = len(clusters[i1]), len(clusters[i2])
		
		if m == 1 and n == 1:
			S.append([dist(clusters[i1][0], clusters[i2][0]), i1, i2])
		else:
			S2 = np.zeros((m,n))
			for i in range(m):
				for j in range(n):
					S2[i][j] = dist( (clusters[i1][i]), clusters[i2][j] )
			
			S.append([np.amax(S2), i1, i2])		# maximum distance, clusters id
	minID = np.argmin(np.array(S)[:,0])			# closest clusters
	return S[minID][0], S[minID][1], S[minID][2]
	
def averageLinkage(clusters):
	S = []
	comb_ids = list( combinations( range(len(clusters)), 2 ) )
	for i1, i2 in comb_ids:
		m, n = len(clusters[i1]), len(clusters[i2])
		
		if m == 1 and n == 1:
			S.append([dist(clusters[i1][0], clusters[i2][0]), i1, i2])
		else:
			S2 = np.zeros((m,n))
			for i in range(m):
				for j in range(n):
					S2[i][j] = dist( (clusters[i1][i]), clusters[i2][j] )
			
			S.append([np.mean(S2), i1, i2])		# average distance, clusters id
	minID = np.argmin(np.array(S)[:,0])			# closest clusters
	return S[minID][0], S[minID][1], S[minID][2]

def centroidLinkage(clusters):
	S = []
	comb_ids = list( combinations( range(len(clusters)), 2 ) )
	for i1, i2 in comb_ids:
		mean1= [ np.mean(col) for col in zip(*clusters[i1]) ]
		mean2= [ np.mean(col) for col in zip(*clusters[i2]) ]
		S.append([dist(mean1,mean2), i1, i2])	# clusters mean distance, clusters id
	minID = np.argmin(np.array(S)[:,0])
	return S[minID][0], S[minID][1], S[minID][2]

def hierarchical(DATA):
	X = copy.deepcopy(DATA)
	
	clusterings = []							# list of all clusterings
	clusters_x  = [ [x] for x in X]				# initial clusters
	clusters_i  = [ [i] for i,x in enumerate(X)]# data points ids in each cluster

	##################################################### initial clustering
	counter=0
	clusterings.append({'n_level':		int(counter), 		# dictionary of first clustering (each data point is a cluster)
						'n_clusters':	len(clusters_x),
						'clusters_x':	clusters_x[:], 
						'clusters_i':	clusters_i[:]
						})	
	print('==> Level %d\t includes %d clusters' %(counter, len(clusters_x)))


	##################################################### generate hierarchical
	while len(clusters_x) > 1:
		counter+=1

		d,id1,id2  = averageLinkage(clusters_x)		# 'singleLinkage', completeLinkage, averageLinkage, centriodLinkage 
		new_cluster_x  = clusters_x[id1] + clusters_x[id2]
		new_cluster_i  = clusters_i[id1] + clusters_i[id2]
		
		clusters_x  = [new_cluster_x] + clusters_x[0:min(id1,id2)] + clusters_x[min(id1,id2)+1: max(id1,id2)] + clusters_x[max(id1,id2)+1: len(clusters_x)]
		clusters_i  = [new_cluster_i] + clusters_i[0:min(id1,id2)] + clusters_i[min(id1,id2)+1: max(id1,id2)] + clusters_i[max(id1,id2)+1: len(clusters_i)]
		
		clusterings.append({
						'n_level': 	 	int(counter), 		# storing resulted clustering as a new dictionary
						'n_clusters':	len(clusters_x),
						'clusters_x':	clusters_x[:],
						'clusters_i':	clusters_i[:]
						})
		print('==> Level %d\t includes %d clusters' %(counter, len(clusters_x)))
	return clusterings

def labels(X, clusters_i):
	labels = []
	for i in range(len(X)):
		for k, cluster in enumerate(clusters_i):
			for idx in cluster:
				if idx == i: labels.append(k); break
			if len(labels) == i+1: break
	return labels

def centers(clusters):
	C = [ [] for cluster in clusters ]
	for i, cluster in enumerate(clusters):
		C[i] = [ np.mean(col) for col in zip(*cluster) ]
	return C
	
##################################################### Run the program

#Artificial data
DATA = prepareData()
plt.scatter(DATA[:,0], DATA[:,1])
plt.show()

'''
#Image data
image = "img4.bmp"
DATA, imRow, imCol, imDim = prepare_image(image)
'''
clusteringsDictionary = hierarchical(DATA)

while True:
	try:
		numberOfClusters = str(input('Enter the number of clusters you want to view:'))
		dictionary 		 = [ dict for dict in clusteringsDictionary if dict['n_clusters'] == int(numberOfClusters)]
		
		
		clustersToHandle = dictionary[0]['clusters_x']
		for i, clust in enumerate(clustersToHandle):
			F1, F2 = zip(*clust)
			plt.scatter( F1, F2)
		plt.show()
		
		'''
		clusters_x = dictionary[0]['clusters_x']
		clusters_i = dictionary[0]['clusters_i']
		
		C = centers(clusters_x)
		Y = labels(DATA,clusters_i)
		
		clusteredImage = np.array([ C[i] for i in Y ])
		display_Image(DATA, clusteredImage, imRow, imCol, imDim)
		'''
	except ValueError:
		if numberOfClusters == 'q': print("\nProgram is ended"); break
		print("Invalid number of clusters")