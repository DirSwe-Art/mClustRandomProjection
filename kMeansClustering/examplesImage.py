'''
Dirar Sweidan
Created:        2024_08_20
Last modified:  2024_  _

- This file containes clustering examples of RGB pixel image data.
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from kMeansFromScratch import kmeans
from sklearn.cluster import KMeans
from PIL import Image

# =============================================================== definitions

def prepare_data(image):
	#IMG = plt.imread(image)	# uint8 data type
	IMG = np.array( Image.open(image, formats=['bmp']) )
	
	imRow, imCol, imDim = IMG.shape
	X = []
	
	for r in range(imRow):
		for c in range(imCol):
			X.append( IMG[r][c] )
			
	return X, imRow, imCol, imDim

def display_image(X, XX, imRow, imCol, imDim):
	X  = np.array(X, dtype='uint8')
	XX = np.array(XX, dtype='uint8')
	IMG_X  = X.reshape(imRow, imCol, imDim)
	IMG_XX = XX.reshape(imRow, imCol, imDim)
	
	plt.close('all')
	f, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True)
	ax1.imshow(IMG_X)
	ax1.set_title('Original Image')
	
	ax2.imshow(IMG_XX)
	ax2.set_title('Segmented Image')
	
	plt.xticks([])
	plt.yticks([])
	plt.savefig('segmented_'+str(k)+'.jpg')
	plt.show()
	

# ================================================================== Clustering X

k     = 2
eps   = 0.0001
image = 'img.bmp' # 'img.jpg'

X, imRow, imCol, imDim = prepare_data(image)

C, L, Y = kmeans(X, k, eps)
XX = np.array( [C[i] for i in Y] ) # replace each data point with its cluster's center (color)

display_image(X, XX, imRow, imCol, imDim)


# ================================================================== Clustering X

cl    = KMeans(n_clusters=k)
cl.fit(X)

Y     = cl.predict(X)
centers = [ center for center in cl.cluster_centers_]
XX    = np.array( [centers[i] for i in Y] )

display_image(X, XX, imRow, imCol, imDim)