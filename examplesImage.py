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


# =============================================================== definitions

def prepare_data(image):
	IMG = mpimg.imread(image)	# uint8 data type
	imRow, imCol, imDim = IMG.shape
	X = []
	
	for r in range(imRow):
		for c in range(imCol):
			X.append( IMG[r][c] )
			
	X = np.array(X, dtype='float32')
	return X, imRow, imCol, imDim

def disply_image(X, XX, imRow, imCol, imDim):
	X = np.array(X, dtype='uint8')
	X = np.array(XX, dtype='uint8')
	IMG_X  = X.reshape(imRow, imCol, imDim)
	IMG_XX = X.reshape(imRow, imCol, imDim)
	
	plt.close('all')
	f, (ax1, ax2) = plt.subplots(2,1, sharex=True, shrey=True)
	ax1.imshow(IMF_X)
	ax1.set_title('Original Image')
	
	ax2.imshow(IMF_XX)
	ax2.set_title('Segmented Image')
	
	plt.xticks([])
	plt.yticks([])
	plt.savefig('segmented_'+str(k)+'.jpg')
	plt.show()
	

# ================================================================== Clustering X

k     = 2
eps   = 0.1
image = 'img.bmp'

X, imRow, imCol, imDim = prepare_data(image)
C, L, Y = kmeans(X, k, eps)
XX = np.array( [C[i] for i in Y] ) # replace each data point with its cluster's center (color)

disply_image(X, XX, imRow, imCol, imDim)