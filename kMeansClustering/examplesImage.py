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
#from PIL import Image

# =============================================================== definitions

def prepare_data(image):
	import os
	if not os.path.exists('results'): os.makedirs('results')
	
	IMG = plt.imread(image)	# uint8 data type
	#IMG = np.array( Image.open(image, formats=['bmp']) )
	
	imRow, imCol, imDim = IMG.shape
	X = []
	
	for r in range(imRow):
		for c in range(imCol):
			X.append( IMG[r][c] )
	
	return X, imRow, imCol, imDim

def display_image(X, XX, imRow, imCol, imDim, text=None):
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
	plt.savefig(r'results/'+text+'segmented_'+str(k)+'.jpg')
	

# ================================================================== Clustering X

k     = 2
eps   = 0.0001
image = 'source_images/img.bmp' # 'img.jpg'

X, imRow, imCol, imDim = prepare_data(image)

C, L, Y = kmeans(X, k, eps)
XX = np.array( [C[i] for i in Y] ) # replace each data point with its cluster's center (color)

display_image(X, XX, imRow, imCol, imDim, text='mine_')


#Image.fromarray(np.uint8(X.reshape(imRow, imCol, imDim))).show()
#Image.fromarray(np.uint8(XX.reshape(imRow, imCol, imDim))).show()

# ================================================================== Clustering X

cl    = KMeans(n_clusters=k)
cl.fit(X)

Y     = cl.predict(X)
centers = [ center for center in cl.cluster_centers_]
XX    = np.array( [centers[i] for i in Y] )

display_image(X, XX, imRow, imCol, imDim, text='lib_')









'''
When I centered the image, colors are changed. However, got similar clustering results.
X = np.array(X)					# Center data if already uncentered
X = X - X.mean(axis=0)
	
	

plt.close('all')
f, ax = plt.subplots()
ax.imshow(IMG_SEG)
ax.set_title('Segmented image')
plt.show()

#############

plt.close('all')
f, axarr = plt.subplots(2,sharex=True)
axarr[0].imshow(IMG)
axarr[0].set_title('Original image')
axarr[1].imshow(IMG_SEG)
axarr[1].set_title('Segmented image')
plt.show()

#############

plt.close('all')
f, (ax1, ax2) = plt.subplots(1,2,sharey=True, sharex=True)
ax1.imshow(IMG)
ax1.set_title('Original image')
ax2.imshow(IMG_SEG)
ax2.set_title('Segmented image')
plt.show()

#############



'''