''' adapted from https://github.com/all-umass/ManifoldWarping '''

import numpy as np
import scipy as sp
import sys
import time
import scipy.spatial.distance as sd
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.manifold import Isomap,LocallyLinearEmbedding
import pandas as pd
try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip
from itertools import product
from functools import reduce
import random
from matplotlib import pyplot


''' Inter-data correspondences '''
class Correspondence(object):

  def __init__(self, pairs=None, matrix=None):
    assert pairs is not None or matrix is not None, \
      'Must provide either pairwise or matrix correspondences'
    self._pairs = pairs
    self._matrix = matrix

  def pairs(self):
    if self._pairs is None:
      self._pairs = np.vstack(np.nonzero(self._matrix)).T
    return self._pairs

  def matrix(self):
    if self._matrix is None:
      self._matrix = np.zeros(self._pairs.max(axis=0)+1)
      for i in self._pairs:
        self._matrix[i[0],i[1]] = 1
    return self._matrix

  def dist_from(self, other):
    '''Calculates the warping path distance from this correspondence to another.
       Based on the implementation from CTW.'''
    B1, B2 = self._bound_row(), other._bound_row()
    gap0 = np.abs(B1[:-1,1] - B2[:-1,1])
    gap1 = np.abs(B1[1:,0] - B2[1:,0])
    d = gap0.sum() + (gap0!=gap1).sum()/2.0
    return d / float(self.pairs()[-1,0]*other.pairs()[-1,0])

  def warp(self, A, XtoY=True):
    '''Warps points in A by pairwise correspondence'''
    P = self.pairs()
    if not XtoY:
      P = np.fliplr(P)
    warp_inds = np.zeros(A.shape[0],dtype=np.int)
    j = 0
    for i in range(A.shape[0]):
      while P[j,0] < i:
        j += 1
      warp_inds[i] = P[j,1]
    return A[warp_inds]

  def _bound_row(self):
    P = self.pairs()
    n = P.shape[0]
    B = np.zeros((P[-1,0]+1,2),dtype=np.int)
    head = 0
    while head < n:
      i = P[head,0]
      tail = head+1
      while tail < n and P[tail,0] == i:
        tail += 1
      B[i,:] = P[(head,tail-1),1]
      head = tail
    return B

'''Distance functions, grouped by metric.'''

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
    
class SparseL2Metric(Metric):
  '''scipy.spatial.distance functions don't support sparse inputs,
  so we have a separate SparseL2 metric for dealing with them'''
  def __init__(self):
    Metric.__init__(self, euclidean_distances, 'sparseL2')

  def within(self, A):
    return euclidean_distances(A,A)

  def between(self,A,B):
    return euclidean_distances(A,B)

  def pairwise(self,A,B):
    '''distances between pairs of rows in A and B'''
    return Metric.pairwise(self, A, B).flatten()

# commonly-used metrics
L1 = Metric(sd.cityblock,'cityblock')
L2 = Metric(sd.euclidean,'euclidean')
SquaredL2 = Metric(sd.sqeuclidean,'sqeuclidean')
SparseL2 = SparseL2Metric()

def dtw(X, Y, metric=SquaredL2, debug=False):
  '''Dynamic Time Warping'''
  dist = metric.between(X,Y)
  if debug:
    path = _python_dtw_path(dist)
  else:
    path = _dtw_path(dist)
  return Correspondence(pairs=path)


def _python_dtw_path(dist):
  '''Pure python, slow version of DTW'''
  nx,ny = dist.shape
  cost = np.zeros(dist.shape)
  trace = np.zeros(dist.shape,dtype=np.int)
  cost[0,:] = np.cumsum(dist[0,:])
  cost[:,0] = np.cumsum(dist[:,0])
  trace[0,:] = 1
  trace[:,0] = 0
  for i,j in product(range(1,nx),range(1,ny)):
    choices = dist[i,j] + np.array((cost[i-1,j], cost[i,j-1], cost[i-1,j-1]))
    trace[i,j] = choices.argmin()
    cost[i,j] = choices.min()
  path = [(nx-1,ny-1)]
  while not (i == 0 and j == 0):
    s = trace[i,j]
    if s == 0:
      i -= 1
    elif s == 1:
      j -= 1
    else:
      i -= 1
      j -= 1
    path.append((i,j))
  return np.array(path)[::-1]


# Shenanigans for running the fast C version of DTW,
# but falling back to pure python if needed
try:
  from scipy.weave import inline
  from scipy.weave.converters import blitz
except ImportError:
  _dtw_path = _python_dtw_path
else:
  def _dtw_path(dist):
    '''Fast DTW, with inlined C'''
    nx,ny = dist.shape
    path = np.zeros((nx+ny,2),dtype=np.int)
    code = '''
    int i,j;
    double* cost = new double[ny];
    cost[0] = dist(0,0);
    for (j=1; j<ny; ++j) cost[j] = dist(0,j) + cost[j-1];
    char** trace = new char*[nx];
    for (i=0; i<nx; ++i) {
      trace[i] = new char[ny];
      trace[i][0] = 0;
    }
    for (j=0; j<ny; ++j) {
      trace[0][j] = 1;
    }
    double diag,c;
    for (i=1; i<nx; ++i){
      diag = cost[0];
      cost[0] += dist(i,0);
      for (j=1; j<ny; ++j){
        // c <- min(cost[j],cost[j-1],diag), trace <- argmin
        if (diag < cost[j]){
          if (diag < cost[j-1]){
            c = diag;
            trace[i][j] = 2;
          } else {
            c = cost[j-1];
            trace[i][j] = 1;
          }
        } else if (cost[j] < cost[j-1]){
          c = cost[j];
          trace[i][j] = 0;
        } else {
          c = cost[j-1];
          trace[i][j] = 1;
        }
        diag = cost[j];
        cost[j] = dist(i,j) + c;
      }
    }
    delete[] cost;
    i = nx-1;
    j = ny-1;
    int p = nx+ny-1;
    for (;p>=0; --p){
      path(p,0) = i;
      path(p,1) = j;
      if (i==0 && j==0) break;
      switch (trace[i][j]){
        case 0: --i; break;
        case 1: --j; break;
        default: --i; --j;
      }
    }
    for (i=0; i<nx; ++i) delete[] trace[i];
    delete[] trace;
    return_val = p;
    '''
    p = inline(code,('nx','ny','dist','path'),type_converters=blitz)
    return path[p:]

''' miscellaneous utilities '''


def pairwise_error(A,B,metric=L2):
  ''' sum of distances between points in A and B, normalized '''
  return metric.pairwise(A/A.max(),B/B.max()).sum()


def block_antidiag(*args):
  ''' makes a block anti-diagonal matrix from the block matices given '''
  return np.fliplr(sp.linalg.block_diag(*map(np.fliplr,args)))


class Timer(object):
  '''Context manager for simple timing of code:
  with Timer('test 1'):
    do_test1()
  '''
  def __init__(self, name, out=sys.stdout):
    self.name = name
    self.out = out

  def __enter__(self):
    self.start = time.time()

  def __exit__(self,*args):
    self.out.write("%s : %0.3f seconds\n" % (self.name, time.time()-self.start))
    return False


def neighbor_graph(X, metric=SquaredL2, k=None, epsilon=None, symmetrize=True):
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


def laplacian(W, normed=False, return_diag=False):
  '''Same as the dense laplacian from scipy.sparse.csgraph'''
  n_nodes = W.shape[0]
  lap = -np.asarray(W)  # minus sign leads to a copy
  # set diagonal to zero, in case it isn't already
  lap.flat[::n_nodes + 1] = 0
  d = -lap.sum(axis=0)  # re-negate to get positive degrees
  if normed:
    d = np.sqrt(d)
    d_zeros = (d == 0)
    d[d_zeros] = 1  # avoid div by zero
    # follow is the same as: diag(1/d) x W x diag(1/d) (where x is np.dot)
    lap /= d
    lap /= d[:, np.newaxis]
    lap.flat[::n_nodes + 1] = 1 - d_zeros
  else:
    # put the degrees on the diagonal
    lap.flat[::n_nodes + 1] = d
  if return_diag:
    return lap, d
  return lap


def lapeig(W=None, L=None, num_vecs=None, return_vals=False):
  tmp_L = (L is None)  # we can overwrite L if it's a tmp variable
  if L is None:
    L = laplacian(W, normed=True)
  vals,vecs = sp.linalg.eigh(L, overwrite_a=tmp_L)  # assumes L is symmetric!
  # not guaranteed to be in sorted order
  idx = np.argsort(vals)
  vecs = vecs.real[:,idx]
  vals = vals.real[idx]
  # discard any with really small eigenvalues
  for i in xrange(vals.shape[0]):
    if vals[i] >= 1e-8:
      break
  if num_vecs is None:
    # take all of them
    num_vecs = vals.shape[0] - i
  embedding = vecs[:,i:i+num_vecs]
  if return_vals:
    return embedding, vals[i:i+num_vecs]
  return embedding


def lapeig_linear(X=None,W=None,L=None,num_vecs=None,k=None,eball=None):
  if L is None:
    if W is None:
      W = neighbor_graph(X, k=k, epsilon=eball)
    L = laplacian(W)
  u,s,_ = np.linalg.svd(np.dot(X.T,X))
  Fplus = np.linalg.pinv(np.dot(u,np.diag(np.sqrt(s))))
  T = reduce(np.dot,(Fplus,X.T,L,X,Fplus.T))
  L = 0.5*(T+T.T)
  return lapeig(L=L,num_vecs=num_vecs)


def isomap(X=None,W=None,num_vecs=None,k=None):
  embedder = Isomap(n_neighbors=k, n_components=num_vecs)
  return embedder.fit_transform(X)


def lle(X=None,W=None,num_vecs=None,k=None):
  embedder = LocallyLinearEmbedding(n_neighbors=k, n_components=num_vecs)
  return embedder.fit_transform(X)


def slow_features(X=None,num_vecs=None):
  assert X.shape[0] >= 2, 'Must have at least 2 points to compute derivative'
  t_cov = np.cov(X, rowvar=False)  # variables are over columns
  dXdt = np.diff(X, axis=0)
  dt_cov = np.cov(dXdt, rowvar=False)
  if num_vecs is not None:
    num_vecs = (0,num_vecs-1)
  vals, vecs = sp.linalg.eigh(dt_cov, t_cov, eigvals=num_vecs, overwrite_a=True, overwrite_b=True)
  return vecs


''' Alignment techniques '''


def _manifold_setup(Wx,Wy,Wxy,mu):
  Wxy = mu * (Wx.sum() + Wy.sum()) / (2 * Wxy.sum()) * Wxy
  W = np.asarray(np.bmat(((Wx,Wxy),(Wxy.T,Wy))))
  return laplacian(W)


def _manifold_decompose(L,d1,d2,num_dims,eps,vec_func=None):
  vals,vecs = np.linalg.eig(L)
  idx = np.argsort(vals)
  for i in range(len(idx)):
    if vals[idx[i]] >= eps:
      break
  vecs = vecs.real[:,idx[i:]]
  if vec_func:
    vecs = vec_func(vecs)
  for i in range(vecs.shape[1]):
    vecs[:,i] /= np.linalg.norm(vecs[:,i])
  map1 = vecs[:d1,:num_dims]
  map2 = vecs[d1:d1+d2,:num_dims]
  return map1,map2


def _linear_decompose(X,Y,L,num_dims,eps):
  Z = sp.linalg.block_diag(X.T,Y.T)
  u,s,_ = np.linalg.svd(np.dot(Z,Z.T))
  Fplus = np.linalg.pinv(np.dot(u,np.diag(np.sqrt(s))))
  T = reduce(np.dot,(Fplus,Z,L,Z.T,Fplus.T))
  L = 0.5*(T+T.T)
  d1,d2 = X.shape[1],Y.shape[1]
  return _manifold_decompose(L,d1,d2,num_dims,eps,lambda v: np.dot(Fplus.T,v))


class LinearAlignment(object):
  def project(self,X,Y,num_dims=None):
    if num_dims is None:
      return np.dot(X,self.pX), np.dot(Y,self.pY)
    return np.dot(X,self.pX[:,:num_dims]), np.dot(Y,self.pY[:,:num_dims])

  def apply_transform(self, other):
    self.pX = np.dot(self.pX, other.pX)
    self.pY = np.dot(self.pY, other.pY)


class TrivialAlignment(LinearAlignment):
  def __init__(self, X, Y, num_dims=None):
    self.pX = np.eye(X.shape[1],num_dims)
    self.pY = np.eye(Y.shape[1],num_dims)


class Affine(LinearAlignment):
  ''' Solves for projection P s.t. Yx = Y*P '''
  def __init__(self, X, Y, corr, num_dims):
    c = corr.pairs()
    assert c.shape[0] > 0, "Can't align data with no correlation"
    Xtrain = X[c[:,0]]
    Ytrain = Y[c[:,1]]
    self.pY = np.linalg.lstsq(Ytrain,Xtrain)[0][:,:num_dims]
    self.pX = np.eye(self.pY.shape[0],num_dims)


class Procrustes:  # note: not a LinearAlignment because it requires mean centering
  ''' Solves for scaling k and rotation Q s.t. Yx = k*Y*Q '''
  def __init__(self, X, Y, corr, num_dims):
    c = corr.pairs()
    Xtrain = X[c[:,0]]
    Ytrain = Y[c[:,1]]
    mX = Xtrain - X.mean(0)
    mY = Ytrain - Y.mean(0)
    u,s,vT = np.linalg.svd(np.dot(mY.T,mX))
    k = s.sum() / np.trace(np.dot(mY.T,mY))
    self.pY = k*np.dot(u,vT)[:num_dims]

  def project(self,X,Y,num_dims=None):
    mX = X - X.mean(0)
    mY = Y - Y.mean(0)
    if num_dims is None:
      return mX, np.dot(mY,self.pY)
    return mX[:,:num_dims], np.dot(mY,self.pY[:,:num_dims])


class CCA(LinearAlignment):
  def __init__(self,X,Y,corr,num_dims,eps=1e-8):
    Wxy = corr.matrix()
    L = laplacian(block_antidiag(Wxy,Wxy.T))
    self.pX, self.pY = _linear_decompose(X,Y,L,num_dims,eps)


class CCAv2:  # same deal as with procrustes
  def __init__(self,X,Y,num_dims):
    mX = X - X.mean(0)
    mY = Y - Y.mean(0)
    Cxx = np.dot(mX.T,mX)
    Cyy = np.dot(mY.T,mY)
    Cxy = np.dot(mX.T,mY)
    d1,d2 = Cxy.shape
    if np.linalg.matrix_rank(Cxx) < d1 or np.linalg.matrix_rank(Cyy) < d2:
      lam = X.shape[0]/2.0
    else:
      lam = 0
    Cx = block_antidiag(Cxy,Cxy.T)
    Cy = sp.linalg.block_diag(Cxx + lam*np.eye(d1), Cyy + lam*np.eye(d2))
    vals,vecs = sp.linalg.eig(Cx,Cy)
    vecs = vecs[np.argsort(vals)[::-1]]  # descending order
    self.pX = vecs[:d1,:num_dims]
    self.pY = vecs[d1:d1+d2,:num_dims]

  def project(self,X,Y,num_dims=None):
    mX = X - X.mean(0)
    mY = Y - Y.mean(0)
    if num_dims is None:
      return np.dot(mX,self.pX), np.dot(mY,self.pY)
    return np.dot(mX,self.pX[:,:num_dims]), np.dot(mY,self.pY[:,:num_dims])


try:
  from sklearn import pls
  pls.CCA  # make sure it exists

  class CCAv3:
    def __init__(self,X,Y,num_dims):
      self._model = pls.CCA(n_components=num_dims)
      self._model.fit(X,Y)

    def project(self,X,Y,num_dims=None):
      pX,pY = self._model.transform(X,Y)
      if num_dims is None:
        return pX,pY
      return pX[:,:num_dims], pY[:,:num_dims]
except ImportError:
  pass


class ManifoldLinear(LinearAlignment):
  def __init__(self,X,Y,corr,num_dims,Wx,Wy,mu=0.9,eps=1e-8):
    L = _manifold_setup(Wx,Wy,corr.matrix(),mu)
    self.pX, self.pY = _linear_decompose(X,Y,L,num_dims,eps)


def manifold_nonlinear(X,Y,corr,num_dims,Wx,Wy,mu=0.9,eps=1e-8):
  L = _manifold_setup(Wx,Wy,corr.matrix(),mu)
  return _manifold_decompose(L,X.shape[0],Y.shape[0],num_dims,eps)




''' Warping aligners '''


def ctw(X,Y,num_dims,metric=SquaredL2,threshold=0.01,max_iters=100,eps=1e-8):
  projecting_aligner = lambda A,B,corr: CCA(A,B,corr,num_dims,eps=eps)
  correlating_aligner = lambda A,B: dtw(A,B,metric=metric)
  return alternating_alignments(X,Y,projecting_aligner,correlating_aligner,threshold,max_iters)


def manifold_warping_linear(X,Y,num_dims,Wx,Wy,mu=0.9,metric=SquaredL2,threshold=0.01,max_iters=100,eps=1e-8):
  projecting_aligner = lambda A,B,corr: ManifoldLinear(A,B,corr,num_dims,Wx,Wy,mu=mu,eps=eps)
  correlating_aligner = lambda A,B: dtw(A,B,metric=metric)
  return alternating_alignments(X,Y,projecting_aligner,correlating_aligner,threshold,max_iters)


def manifold_warping_nonlinear(X,Y,num_dims,Wx,Wy,mu=0.9,metric=SquaredL2,threshold=0.01,max_iters=100,eps=1e-8):
  projecting_aligner = lambda A,B,corr: manifold_nonlinear(A,B,corr,num_dims,Wx,Wy,mu=mu,eps=eps)
  correlating_aligner = lambda A,B: dtw(A,B,metric=metric)
  return alternating_alignments_nonlinear(X,Y,projecting_aligner,correlating_aligner,threshold,max_iters)


def ctw_twostep(X,Y,num_dims,embedder=isomap,**kwargs):
  alt_aligner = lambda A,B,n,**kwargs: ctw(A,B,n,**kwargs)
  return twostep_alignment(X,Y,num_dims,embedder,alt_aligner)


def manifold_warping_twostep(X,Y,num_dims,Wx,Wy,embedder=isomap,**kwargs):
  alt_aligner = lambda A,B,n,**kwargs: manifold_warping_linear(A,B,n,Wx,Wy,**kwargs)
  return twostep_alignment(X,Y,num_dims,embedder,alt_aligner)


def twostep_alignment(X,Y,num_dims,embedder,alt_aligner):
  X_proj, Y_proj = embedder(X,num_dims,k=5), embedder(Y,num_dims,k=5)
  corr, aln = alt_aligner(X_proj,Y_proj,num_dims)
  X_proj, Y_proj = aln.project(X_proj,Y_proj)
  return corr, X_proj, Y_proj


def alternating_alignments(X,Y,proj_align,corr_align,threshold,max_iters):
  corr = Correspondence(pairs=np.array(((0,0),(X.shape[0]-1,Y.shape[0]-1))))
  aln = TrivialAlignment(X,Y)
  X_proj,Y_proj = X.copy(), Y.copy()  # same as aln.project(X,Y)
  for it in range(max_iters):
    aln.apply_transform(proj_align(X_proj,Y_proj,corr))
    X_proj,Y_proj = aln.project(X,Y)
    new_corr = corr_align(X_proj,Y_proj)
    if corr.dist_from(new_corr) < threshold:
      return new_corr, aln
    corr = new_corr
  return corr, aln


def alternating_alignments_nonlinear(X,Y,proj_align,corr_align,threshold,max_iters):
  corr = Correspondence(pairs=np.array(((0,0),(X.shape[0]-1,Y.shape[0]-1))))
  X_proj,Y_proj = proj_align(X,Y,corr)
  for it in range(max_iters):
    new_corr = corr_align(X_proj,Y_proj)
    if corr.dist_from(new_corr) < threshold:
      return new_corr, X_proj, Y_proj
    corr = new_corr
    X_proj,Y_proj = proj_align(X_proj,Y_proj,corr)
  return corr, X_proj, Y_proj


def show_alignment(X,Y,titX=None,titY=None,title=None,legend=True):
  '''plot two data sets on the same figure'''
  dim = X.shape[1]
  assert dim == Y.shape[1], 'dimensionality must match'
  assert dim in (1,2,3), ('can only plot 1, 2, or 3-dimensional data, X has shape %dx%d' % X.shape)
  if dim == 1:
    pyplot.plot(X[:,0],label=titX,alpha=0.5)
    pyplot.plot(Y[:,0],label=titX,alpha=0.5)
  elif dim == 2:
    pyplot.scatter(X[:,0],X[:,1],label=titX,alpha=0.5)
    pyplot.scatter(Y[:,0],Y[:,1],label=titY,alpha=0.5)
  else:  # dim == 3
    from mpl_toolkits.mplot3d import Axes3D
    fig = pyplot.gcf()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],X[:,2],label=titX,alpha=0.5)
    ax.scatter(Y[:,0],Y[:,1],Y[:,2],label=titY,alpha=0.5)
  if title:
    pyplot.title(title)
  if legend:
    pyplot.legend(loc='best')
  return pyplot.show

def show_neighbor_graph(X,corr,title=None,fig=None,ax=None):
  '''Plot the neighbor connections between points in a data set.
     Note: plotting correspondences for 3d points is slow!'''
  assert X.shape[1] in (2,3), 'can only show neighbor graph for 2d or 3d data'
  if X.shape[1] == 2:
    if ax is None:
      ax = pyplot.gca()
    for pair in corr.pairs():
      ax.plot(X[pair,0], X[pair,1], 'r-')
    ax.plot(X[:,0],X[:,1],'o')
  else:
    if ax is None:
      from mpl_toolkits.mplot3d import Axes3D
      if fig is None:
        fig = pyplot.gcf()
      ax = Axes3D(fig)
    for pair in corr.pairs():
      ax.plot(X[pair,0], X[pair,1], X[pair,2], 'r-')
    ax.plot(X[:,0],X[:,1],X[:,2],'o')
  if title:
    ax.set_title(title)
  return pyplot.show



def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C




def ManiNetCluster(X,Y,nameX=None,nameY=None,corr=None,d=3,method='linear manifold',k_NN=5, k_medoids=60):
  # df_in1 = pd.read_csv(file1)
  # df_in2 = pd.read_csv(file2)
  # X = df_in1.as_matrix()[:,1:df_in1.shape[1]]
  # Y = df_in2.as_matrix()[:,1:df_in2.shape[1]]
  
  Wx = neighbor_graph(X, k=k_NN)
  Wy = neighbor_graph(Y, k=k_NN)
  lin_aligners = {
    'no alignment':     (lambda: TrivialAlignment(X,Y)),
    'affine':           (lambda: Affine(X,Y,corr,d)),
    'procrustes':       (lambda: Procrustes(X,Y,corr,d)),
    'cca':              (lambda: CCA(X,Y,corr,d)),
    'cca_v2':           (lambda: CCAv2(X,Y,d)),
    'linear manifold':  (lambda: ManifoldLinear(X,Y,corr,d,Wx,Wy)),
    'ctw':              (lambda: ctw(X,Y,d)[1]),
    'manifold warping': (lambda: manifold_warping_linear(X,Y,d,Wx,Wy)[1])
  }
  other_aligners = {
    'dtw':              (lambda: (X, dtw(X,Y).warp(X))),
    'nonlinear manifold aln':
                        (lambda: manifold_nonlinear(X,Y,corr,d,Wx,Wy)),
    'nonlinear manifold warp':
                        (lambda: manifold_warping_nonlinear(X,Y,d,Wx,Wy)[1:]),
    'manifold warping two-step':
                        (lambda: manifold_warping_twostep(X_normalized, Y_normalized, d, Wx, Wy)[1:])
  }
  # fig = pyplot.figure()
  # with Timer(method):
  
  if method in lin_aligners:
    Xnew, Ynew = lin_aligners[method]().project(X, Y)
  else:
    Xnew, Ynew = other_aligners[method]()
  # print (' sum sq. error =', pairwise_error(Xnew, Ynew, metric=SquaredL2))
  # show_alignment(Xnew, Ynew, title=method)
  # pyplot.draw()
  # pyplot.show()
  # fig.savefig(time.strftime("%Y%m%d-%H%M%S")+'.pdf')W = np.concatenate((Xnew, Ynew), axis=0)
  W = np.concatenate((Xnew, Ynew), axis=0) # Matrix containing both X and Y
  df_W = pd.DataFrame(W)
  df_W = df_W.add_prefix('Val')
  # distance matrix
  D = pairwise_distances(W, metric='euclidean')
  # split into 60 clusters
  M, C = kMedoids(D, k_medoids)

  C_label = np.zeros(X.shape[0]+Y.shape[0])

  for label in C:
    for point_idx in C[label]:
        C_label[point_idx] = label
        
  X_or_Y = np.repeat(np.array([nameX,nameY]), [Xnew.shape[0], Ynew.shape[0]], axis=0)
  df = pd.DataFrame({'module':C_label, 'data':X_or_Y})
  df = pd.concat([df, df_W], axis=1)
  # 'Val1':W[:,0], 'Val2':W[:,1], 'Val3':W[:,2]})#, 'id':df_in1[df_in1.columns[0]].tolist()+df_in2[df_in2.columns[0]].tolist()})
  return df#, C_label#, pairwise_error(Xnew, Ynew, metric=SquaredL2)#, pyplot.show
  
