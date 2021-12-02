import numpy as np
import scipy.spatial.distance as sd
from sklearn.metrics.pairwise import euclidean_distances


class Metric(object):
  def __init__(self,dist,name):
    self.dist = dist  # dist(x,y): distance between two points
    self.name = name

  def within(self,A):
    '''pairwise distances between each pair of rows in A'''
    return sd.squareform(sd.pdist(A,self.name),force='tomatrix')

  def between(self,A,B):
    '''cartesian product distances between pairs of rows in A and B'''
    return sd.cdist(A,B,self.name)

  def pairwise(self,A,B):
    '''distances between pairs of rows in A and B'''
    return np.array([self.dist(a,b) for a,b in izip(A,B)])


def neighbor_graph(X, metric=Metric(sd.sqeuclidean, 'sqeuclidean'), k=None, epsilon=None, symmetrize=True):
    '''Construct an adj matrix from a matrix of points (one per row)'''
    assert (k is None) ^ (epsilon is None), "Must provide `k` xor `epsilon`"
    dist = metric.within(X)
    adj = np.zeros(dist.shape)  # TODO: scipy.sparse support, or at least use a smaller dtype
    if k is not None:
        # do k-nearest neighbors
        nn = np.argsort(dist)[:,:min(k+1,len(X))]
        # nn's first column is the point idx, rest are neighbor idxs
        if symmetrize:
            for inds in nn:
                adj[inds[0],inds[1:]] = 1
                adj[inds[1:],inds[0]] = 1
        else:
            for inds in nn:
                adj[inds[0],inds[1:]] = 1
    else:
        # do epsilon-ball
        p_idxs, n_idxs = np.nonzero(dist<=epsilon)
        for p_idx, n_idx in zip(p_idxs, n_idxs):
            if p_idx != n_idx:  # ignore self-neighbor connections
                adj[p_idx,n_idx] = 1
        # epsilon-ball is typically symmetric, assuming a normal distance metric
    return adj
