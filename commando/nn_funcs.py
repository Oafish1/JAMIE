import numpy as np
from scipy.sparse.csgraph import connected_components
import scipy.spatial.distance as sd
import torch


def knn(data, k=5):
    """Connected KNN generation"""
    # ASDF: Make sparse
    # from scipy.sparse import csr_matrix
    # adj = csr_matrix(adj)
    # Adapted version of ``neighbor_graph`` from ManiNetCluster
    dist = sd.squareform(sd.pdist(data, 'sqeuclidean'), force='tomatrix')
    adj = np.zeros(dist.shape)
    idxs = np.argsort(dist)[:, :k+1]
    for idx in idxs:
        adj[idx[0], idx[1:]] = dist[idx[0], idx[1:]]
        # Symmetrize
        adj[idx[1:], idx[0]] = dist[idx[1:], idx[0]]

    # Connection via
    # https://www.nature.com/articles/s41467-020-16822-4
    n_components, labels = connected_components(adj, directed=False)
    for i in range(n_components-1):
        g1_idx = np.arange(dist.shape[0])[np.array(labels) == i]
        g2_idx = np.arange(dist.shape[0])[np.array(labels) == i+1]
        idx = np.concatenate((g1_idx, g2_idx))
        sub_dist = dist[g1_idx][:, g2_idx]
        min_dist = np.unravel_index(np.argmin(sub_dist, axis=None), sub_dist.shape)
        g1_new_idx = g1_idx[min_dist[0]]
        g2_new_idx = g2_idx[min_dist[1]]
        adj[g1_new_idx, g2_new_idx] = sub_dist[min_dist]
    # assert connected_components(adj, directed=False)[0] == 1, (
    #     'Something went wrong when connecting components...'
    # )

    # ASDF: Implement Gaussian kernel
    # adj = adj.toarray()
    # adj[adj > 0] = 1 / adj[adj > 0]
    # adj[adj > 0] = -np.log(adj[adj > 0])
    adj[adj > 0] = np.exp(-adj[adj > 0])
    return adj


def uc_loss(primes, F, pairwise=False):
    """Select loss term from UnionCom"""
    if pairwise:
        uc_loss = 0
        for i in range(primes[0].shape[0]):
            partial_sum = 0
            for j in range(primes[1].shape[0]):
                partial_sum += primes[1][j] * F[i, j]
            norm = primes[0][i] - partial_sum
            norm = torch.square(norm).sum()
            uc_loss += norm
    else:
        norm = primes[0] - torch.mm(F, primes[1])
        uc_loss = torch.square(norm).sum()
    return uc_loss


def nlma_loss(
    primes,
    Wx,
    Wy,
    Wxy,
    mu,
    fg=True,
    ff=True,
    gg=True,
):
    """Compute NLMA loss"""
    if not (fg and ff and gg):
        nlma_loss = 0
        for i in range(primes[0].shape[0]):
            for j in range(primes[1].shape[0]):
                if fg:
                    norm = primes[0][i] - primes[1][j]
                    norm = torch.square(norm).sum()
                    nlma_loss += norm * Wxy[i, j] * (1 - mu)

                if ff:
                    norm = primes[0][i] - primes[0][j]
                    norm = torch.square(norm).sum()
                    nlma_loss += norm * Wx[i, j] * mu

                if gg:
                    norm = primes[1][i] - primes[1][j]
                    norm = torch.square(norm).sum()
                    nlma_loss += norm * Wy[i, j] * mu
    else:
        # Fast matrix version
        # ASDDF: Get into normal, interprable loss range
        num_cells = Wxy.shape[0]

        Dx = torch.sum(Wx, dim=0)
        Dy = torch.sum(Wy, dim=0)
        D = torch.diag(torch.cat((Dx, Dy), dim=0))
        W = torch.block_diag(Wx, Wy)
        W[:num_cells][:, num_cells:] += Wxy
        W[num_cells:][:, :num_cells] += torch.t(Wxy)

        L = D - W
        P = torch.cat(primes, dim=0)

        nlma_loss = torch.trace(torch.mm(torch.mm(torch.t(P), L), P))
    return nlma_loss


def gw_loss(primes):
    """Calculate Gromov-Wasserstein Distance"""
    # ASDF Implement fast approximation
    assert all(len(primes[0]) == len(p) for p in primes), (
        'Datasets must be aligned'
    )

    num_cells = len(primes[0])
    loss = 0
    for i in range(num_cells):
        for j in range(num_cells):
            set1 = torch.norm(primes[0][i] - primes[0][j])
            set2 = torch.norm(primes[1][i] - primes[1][j])
            loss += torch.square(set1 - set2)
    return loss
