"""Multi-View Clustering via Orthogonalization"""

# Authors: DirarSweidan
# License: DSB 3-Claus

import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import fractional_matrix_power


# ========================================================================



# ================================================================================
# Paper: Z. Qi et al. (2009). A principled and flexible framework for finding alternative clusterings. SIGKDD (pp. 717-726).
