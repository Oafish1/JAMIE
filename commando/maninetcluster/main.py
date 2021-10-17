''' adapted from https://github.com/all-umass/ManifoldWarping '''

import numpy as np
from scipy.cluster.hierarchy import linkage
import pandas as pd
from sklearn import decomposition, preprocessing
import matplotlib
# matplotlib.use("tkagg")
# from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns

from alignment import TrivialAlignment, Affine, Procrustes, CCA, CCAv2, \
    ManifoldLinear, manifold_nonlinear
from correspondence import Correspondence
from distance import SquaredL2
from neighborhood import neighbor_graph
# from synthetic_data import swiss_roll, add_noise, spiral
from util import pairwise_error, Timer
# from viz import show_alignment
from warping import ctw, dtw, manifold_warping_linear, manifold_warping_nonlinear, manifold_warping_twostep

from rpy2.robjects import r
import rpy2.robjects.pandas2ri as pandas2ri

# import sys

file = "expr.RData"
rf = r['load'](file)

dayExpr = pandas2ri.ri2py_dataframe(r['dayExpr'])
nightExpr = pandas2ri.ri2py_dataframe(r['nightExpr'])

X = dayExpr.as_matrix() # datWorm.iloc[:, 0:datWorm.shape[1]].as_matrix()
Y = nightExpr.as_matrix() # datFly.iloc[:, 0:datFly.shape[1]].as_matrix()
n = 17695
d = 3

X_normalized = preprocessing.normalize(X, norm='l2').T
Y_normalized = preprocessing.normalize(Y, norm='l2')[0:13, :].T

corr = Correspondence(matrix=np.eye(n)) # Correspondence(matrix=corr.as_matrix())
Wx = neighbor_graph(X_normalized, k=5)
Wy = neighbor_graph(Y_normalized, k=5)

lin_aligners = (
    ('no alignment', lambda: TrivialAlignment(X_normalized, Y_normalized, d)),
    # ('affine',           lambda: Affine(X,Y,corr,d)),
    # ('procrustes',       lambda: Procrustes(X,Y,corr,d)),
    ('cca', lambda: CCA(X_normalized, Y_normalized, corr, d)),
    # ('cca_v2',           lambda: CCAv2(X,Y,d)),
    ('linear manifold', lambda: ManifoldLinear(X_normalized, Y_normalized, corr, d, Wx, Wy)),
    ('ctw', lambda: ctw(X_normalized, Y_normalized, d)[1]),
    ('manifold warping', lambda: manifold_warping_linear(X_normalized, Y_normalized, d, Wx, Wy)[1]),
)

other_aligners = (
    ('nonlinear manifold aln', lambda: manifold_nonlinear(X_normalized, Y_normalized, corr, d, Wx, Wy)),
    ('nonlinear manifold warp', lambda: manifold_warping_nonlinear(X_normalized, Y_normalized, d, Wx, Wy)[1:]),
    ('manifold warping two-step', lambda: manifold_warping_twostep(X_normalized, Y_normalized, d, Wx, Wy)[1:]),
)

# heatmin = 1
# heatmax = 0

metric = SquaredL2
# disMat = np.empty((0,944784), float)
# heatList = []
W = []

# pp = PdfPages('manifold_17695.pdf')
# pyplot.ion()

for name, aln in lin_aligners:
    pyplot.figure()
    pyplot.clf()
    with Timer(name):
        Xnew, Ynew = aln().project(X_normalized, Y_normalized)    
    print (' sum sq. error =', pairwise_error(Xnew, Ynew, metric=SquaredL2))
    # show_alignment(Xnew, Ynew, 'day', 'night', name)
    # pyplot.draw()
    # pp.savefig()
    
    W.append(np.concatenate((Xnew, Ynew), axis=0))
    # print(W[0].shape)
    
    # disMat = np.vstack((disMat, metric.between(Xnew/Xnew.max(), Ynew/Ynew.max()).flatten()))
    # 
    # heatmin = min(heatmin, metric.between(Xnew, Ynew).min())
    # heatmax = max(heatmax, metric.between(Xnew, Ynew).max())
    # heatList.append(metric.between(Xnew/Xnew.max(), Ynew/Ynew.max()))
    
lin_man_clust = linkage(W[0], 'single')
man_warp_clust = linkage(W[1], 'single')

np.save("W0_10d.npy", W[0])
np.save("W1_10d.npy", W[1])
# np.save("lin_man_clust_10d.npy", lin_man_clust)
# np.save("man_warp_clust_10d.npy", man_warp_clust)
# np.save("no_align.npy", heatList[1])
# np.save("cca.npy", heatList[2])
# np.save("lin_manifold.npy", heatList[0])
# np.save("ctw.npy", heatList[4])
# np.save("manifoldwarping.npy", heatList[1])

# lin_man_mat = np.load('lin_manifold.npy')
# lin_man_clust = linkage(lin_man_mat.flatten(), 'single')
# 
# man_warp_mat = np.load('manifoldwarping.npy')
# man_warp_clust = linkage(man_warp_mat.flatten(), 'single')
# 
# np.save("lin_man_clust.npy", lin_man_clust)
# np.save("man_warp_clust.npy", man_warp_clust)

# for name, aln in other_aligners:
#     pyplot.figure()
#     pyplot.clf()
#     with Timer(name):
#         Xnew, Ynew = aln()
#     print (' sum sq. error =', pairwise_error(Xnew, Ynew, metric=SquaredL2))
#     
#     disMat = np.vstack((disMat, metric.between(Xnew/Xnew.max(), Ynew/Ynew.max()).flatten()))
#     
#     heatmin = min(heatmin, metric.between(Xnew, Ynew).min())
#     heatmax = max(heatmax, metric.between(Xnew, Ynew).max())
#     
#     show_alignment(Xnew, Ynew, 'Worm', 'Fly', name) 
#     
#     heatList.append(metric.between(Xnew/Xnew.max(), Ynew/Ynew.max()))
# 
# i = 0
# for name, aln in lin_aligners:
#     cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=20, as_cmap=True)
#     clutermap_fig = sns.clustermap(heatList[i],
#                               vmin = heatmin, vmax=heatmax, cbar_kws={"ticks":[heatmin,heatmax]},
#                               cmap=cmap)
#     pyplot.title(name)
#     i = i + 1
# for name, aln in other_aligners:
#     cmap = sns.diverging_palette(250, 15, s=75, l=40, sep=20, as_cmap=True)
#     clutermap_fig = sns.clustermap(heatList[i],
#                               vmin = heatmin, vmax=heatmax, cbar_kws={"ticks":[heatmin,heatmax]},
#                               cmap=cmap)
#     pyplot.title(name)
#     i = i + 1
# 
# disDf = pd.DataFrame(disMat.T)
# disDf.columns = ['no alignment',
#                  'cca',
#                  'linear manifold',
#                  'ctw',
#                  'manifold warping',
#                  'nonlinear manifold aln',
#                  'nonlinear manifold warp',
#                  'manifold warping two-step']
# a4_dims = (11.7, 8.27)
# fig, ax = pyplot.subplots(figsize=a4_dims)
# 
# boxplt = sns.boxplot(data = disDf)
# boxplt = boxplt.get_figure()
# boxplt.savefig("boxplot1.pdf")
