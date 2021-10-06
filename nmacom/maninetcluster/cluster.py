import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import kmedoids

W0 = np.load("W0_10d.npy")
# distance matrix
D = pairwise_distances(W0, metric='euclidean')
# split into 60 clusters
M, C = kmedoids.kMedoids(D, 60)

C_label = np.zeros(35390) # 35390 = 17695*2 (number of genes from both networks)

for label in C:
    for point_idx in C[label]:
        C_label[point_idx] = label
        
np.save("kmedoids.npy", C_label.astype(int))
