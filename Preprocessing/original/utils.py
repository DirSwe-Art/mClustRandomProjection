from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.pylab import plt
from matplotlib import colors as mcolors
import numpy as np
import cv2
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------
def dataset_image(filename):
	IMG = cv2.imread(filename, cv2.IMREAD_COLOR)
	imRow, imCol, imDim = IMG.shape
	
	X = []
	for i in range(imRow):
		for j in range(imCol):
			X.append( np.array([ float(v) for v in IMG[i][j] ]) )
			# X.append( np.array([ float(v) for v in IMG[i][j] ] + [ i,j ]) )
	
	X = np.array(X)
	# X = MinMaxScaler().fit_transform(X)
	
	return X, IMG.shape
	
# --------------------------------------------
def show_image(centers, Y, image_shape):
	list_to_uint8 = lambda A: [np.uint8(v) for v in A]
	
	# centers = [ list_to_uint8(center) for center in centers ]
	centers = [ list_to_uint8(np.random.randint(0, 255, 3)) for center in centers ]
	
	XX = np.array([ centers[i] for i in Y ])
	IMG_SEG = XX.reshape( image_shape ) # Converting XX into an image (just to show it)
	cv2.imshow('segmented image', IMG_SEG)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# --------------------------------------------
def dataset_artificial(): # Returns a dataset with 4 gaussians
	X = []
	for center in [ [0,2], [0,4], [2.5,2], [2.5,4] ] :
		X += np.random.multivariate_normal(center, np.identity(2)/3, size = 200).tolist()
	return np.array(X)
	
# --------------------------------------------
def plot_data(X, Y=None, transform="PCA"):
	color_names = ['r', 'g', 'b', 'm', 'c', 'y', 'k'] + list( mcolors.CSS4_COLORS.keys() )
	markers = "*.x,ovP1<>pX+2hD^3sHd4|_8"*10
	
	if len(X[0]) > 2:
		XX = PCA(n_components=2).fit_transform(X) if transform=="PCA" else TSNE(n_components=2).fit_transform(X)
	else:
		XX = X
	if Y is None: plt.scatter( *zip(*XX) )
	else: plt.scatter( *zip(*XX), c=[color_names[y] for y in Y] )
	plt.show()
	
# --------------------------------------------
def plot_dendrogram(model, **kwargs):
	children = model.children_
	distance = np.arange(children.shape[0])
	no_of_observations = np.arange(2, children.shape[0]+2)
	linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
	dendrogram(linkage_matrix, **kwargs)
	plt.show()
	
# --------------------------------------------
