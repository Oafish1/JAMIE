from itertools import product
import warnings

import anndata as ad
import numpy as np
from scipy import linalg, stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import torch
import unioncom.UnionCom as uc
from unioncom.utils import (
    geodesic_distances,
    init_random_seed,
    joint_probabilities,
)

from .utilities import time_logger


class ComManDo(uc.UnionCom):
    """Adaptation of https://github.com/caokai1073/UnionCom by caokai1073"""
    def __init__(
        self,
        **kwargs
    ):
        if 'project_mode' not in kwargs:
            kwargs['project_mode'] = 'nlma'
        if 'distance_mode' not in kwargs:
            kwargs['distance_mode'] = 'spearman'

        super().__init__(**kwargs)

    def fit_transform(self, dataset=None):
        """Fit function with ``nlma`` added"""
        distance_modes = [
            # Pairwise
            'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis',
            'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming',
            'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean',
            'haversine',
            # Non-pairwise
            'geodesic', 'spearman', 'pearson',
        ]

        if self.integration_type not in ['MultiOmics']:
            raise Exception('integration_type error! Enter MultiOmics.')
        if self.distance_mode not in distance_modes:
            raise Exception('distance_mode error! Enter a correct distance_mode.')
        if self.project_mode not in ('nlma', 'tsne'):
            raise Exception("Choose correct project_mode: 'nlma', 'tsne'.")
        if self.integration_type != 'MultiOmics':
            raise Exception(
                "ComManDo is only compatible with integration_type: 'MultiOmics'."
            )

        time = time_logger()
        init_random_seed(self.manual_seed)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Test for dataset type (must all be the same)
        self.dataset = dataset
        self.dataset_annotation = None
        if isinstance(self.dataset[0], ad._core.anndata.AnnData):
            self.dataset = [d.X for d in self.dataset]
            self.dataset_annotation = dataset

        self.dataset_num = len(self.dataset)
        for i in range(self.dataset_num):
            self.row.append(np.shape(self.dataset[i])[0])
            self.col.append(np.shape(self.dataset[i])[1])

        if self.project_mode == 'nlma' and not (np.array(self.row) == self.row[0]).all():
            raise Exception("project_mode: 'nlma' requres aligned datasets.")

        # Compute the distance matrix
        self.compute_distances()
        time.log('Distance')

        # Find correspondence between samples
        match_result = self.match()
        pairs_x = []
        pairs_y = []
        for i in range(self.dataset_num - 1):
            cost = np.max(match_result[i]) - match_result[i]
            row_ind, col_ind = linear_sum_assignment(cost)
            pairs_x.append(row_ind)
            pairs_y.append(col_ind)
        time.log('Correspondence')

        #  Project to common embedding
        if self.project_mode == 'tsne':
            P_joint = []
            for i in range(self.dataset_num):
                P_joint.append(joint_probabilities(self.dist[i], self.perplexity))

            for i in range(self.dataset_num):
                if self.col[i] > 50:
                    self.dataset[i] = PCA(n_components=50).fit_transform(self.dataset[i])
                    self.col[i] = 50

            integrated_data = self.project_tsne(self.dataset, pairs_x, pairs_y, P_joint)
        elif self.project_mode == 'nlma':
            match_matrix = [
                [None for j in range(self.dataset_num)]
                for i in range(self.dataset_num)
            ]
            k = 0
            for i, j in product(*(2 * [range(self.dataset_num)])):
                if i == j:
                    # mat, _ = stats.spearmanr(self.dataset[i], axis=1)
                    mat = np.eye(self.row[0])
                elif i > j:
                    mat = match_matrix[j][i].T
                else:
                    mat = match_result[k]
                    k += 1
                match_matrix[i][j] = mat

            integrated_data = self.project_nlma(match_matrix)
        time.log('Mapping')

        print('-' * 33)
        print('ComManDo Done!')
        time.aggregate()

        return integrated_data

    def match(self):
        """Find correspondence between multi-omics datasets"""
        print('Device:', self.device)
        cor_pairs = []
        for i in range(self.dataset_num):
            for j in range(i, self.dataset_num):
                if i == j:
                    continue

                print('-' * 33)
                print(f'Find correspondence between Dataset {i + 1} and Dataset {j + 1}')
                F = self.Prime_Dual(
                    [self.dist[i], self.dist[j]],
                    dx=self.col[i],
                    dy=self.col[j],
                )
                cor_pairs.append(F)

        print("Finished Matching!")
        return cor_pairs

    def fast_Prime_Dual(self, dist, dx=None, dy=None):
        """New home of two-step"""
        pass

    def Prime_Dual(
        self,
        dist,
        dx=None,
        dy=None,
        verbose=True,
    ):
        """Prime dual combined with Adam algorithm to find the local optimal solution"""
        Kx = dist[0]
        Ky = dist[1]

        # Escape for 1x1 data
        if Kx.shape == (1, 1) and Ky.shape == (1, 1):
            warnings.warn('1x1 distance matrix, escaping...')
            return torch.ones((1, 1)).float().to(self.device)

        N = np.int(np.maximum(Kx.shape[0], Ky.shape[0]))
        Kx = Kx / N
        Ky = Ky / N
        Kx = torch.from_numpy(Kx).float().to(self.device)
        Ky = torch.from_numpy(Ky).float().to(self.device)
        a = np.sqrt(dy/dx)
        m = np.shape(Kx)[0]
        n = np.shape(Ky)[0]

        F = torch.zeros((m, n)).float().to(self.device)
        Im = torch.ones((m, 1)).float().to(self.device)
        In = torch.ones((n, 1)).float().to(self.device)
        Inn = torch.ones((n, n)).float().to(self.device)
        Lambda = torch.zeros((n, 1)).float().to(self.device)
        Mu = torch.zeros((m, 1)).float().to(self.device)
        S = torch.zeros((n, 1)).float().to(self.device)

        pho1 = 0.9
        pho2 = 0.999
        delta = 10e-8
        Fst_moment = torch.zeros((m, n)).float().to(self.device)
        Snd_moment = torch.zeros((m, n)).float().to(self.device)

        i = 0
        while(i < self.epoch_pd):
            # Simplified grad with casting (atol=2e-6)
            # 100k iterations: 5.52e-5, 5.48e-5, 5.50e-5
            FKy = torch.mm(F, Ky)
            grad = (
                4 * torch.mm(FKy, torch.mm(torch.t(F), FKy))
                - 4 * a * torch.mm(Kx, FKy)
                + torch.mm(Mu, torch.t(In))
                + torch.mm(Im, torch.t(Lambda))
                + self.rho * (
                    torch.mm(F, Inn)
                    + torch.mm(
                        Im,
                        # Using premade Imm slows computation
                        torch.mm(torch.t(Im), F)
                        + torch.t(S - 2 * In)
                    )
                )
            )

            i += 1
            Fst_moment = pho1 * Fst_moment + (1 - pho1) * grad
            Snd_moment = pho2 * Snd_moment + (1 - pho2) * grad * grad
            hat_Fst_moment = Fst_moment / (1 - np.power(pho1, i))
            hat_Snd_moment = Snd_moment / (1 - np.power(pho2, i))
            grad = hat_Fst_moment / (torch.sqrt(hat_Snd_moment) + delta)
            F_tmp = F - grad
            F_tmp[F_tmp < 0] = 0

            # update
            F = (1 - self.epsilon) * F + self.epsilon * F_tmp

            # update slack variable
            grad_s = Lambda + self.rho*(torch.mm(torch.t(F), Im) - In + S)
            s_tmp = S - grad_s
            s_tmp[s_tmp < 0] = 0
            S = (1-self.epsilon)*S + self.epsilon*s_tmp

            # update dual variables
            Mu = Mu + self.epsilon*(torch.mm(F, In) - Im)
            Lambda = Lambda + self.epsilon*(torch.mm(torch.t(F), Im) - In + S)

            # if scaling factor changes too fast, we can delay the update
            if self.integration_type == "MultiOmics":
                if i >= self.delay:
                    a = (
                        torch.trace(torch.mm(Kx, torch.mm(torch.mm(F, Ky), torch.t(F))))
                        / torch.trace(torch.mm(Kx, Kx))
                    )

            if verbose and i % self.log_pd == 0:
                if self.integration_type == "MultiOmics":
                    norm2 = torch.norm(a*Kx - torch.mm(torch.mm(F, Ky), torch.t(F)))
                    print("epoch:[{:d}/{:d}] err:{:.4f} alpha:{:.4f}"
                          .format(i, self.epoch_pd, norm2.data.item(), a))
                else:
                    norm2 = torch.norm(dist*F)
                    print("epoch:[{:d}/{:d}] err:{:.4f}"
                          .format(i, self.epoch_pd, norm2.data.item()))

        return F.cpu().detach().numpy()

    def project_nlma(self, W, compressed=False):
        """
        Projects using `F` matrices as correspondence using NLMA, heavily
        relies on methodology and code from
        https://github.com/daifengwanglab/ManiNetCluster
        """
        print('-' * 33)
        print('Performing NLMA')
        mu = .9
        eps = 1e-8
        vec_func = None

        print('Applying Coefficients')
        diag_sum = sum(W[i][i] for i in range(self.dataset_num))
        for i, j in (
            (i, j) for i in range(self.dataset_num) for j in range(i+1, self.dataset_num)
        ):
            # NOTE: Assumes that transposes are the same array in memory
            W[i][j] = mu * diag_sum / (self.dataset_num * W[i][j].sum()) * W[i][j]
        W = np.asarray(np.bmat(W))
        # print(W)

        print('Computing Laplacian')
        # Dense calculation of scipy.sparse.csgraph.laplacian from
        # ManiNetCluster
        n_nodes = self.row[0]
        L = -W
        L.flat[::n_nodes + 1] = 0
        d = -L.sum(axis=0)
        L.flat[::n_nodes + 1] = d

        print('Calculating eigenvectors')
        # vals, vecs = linalg.eigh(
        #     L,
        #     lower=False,
        #     overwrite_a=True,
        #     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
        #     # Note: Default of 'evx' is only best when taking very few eigenvectors
        #     # from a large matrix
        #     driver='evx',
        #     subset_by_value=(eps, np.inf),
        # )
        vals, vecs = np.linalg.eig(L)

        idx = np.argsort(vals)
        for i in range(len(idx)):
            if vals[idx[i]] >= eps:
                break
        # print(vals[idx[i:]])
        vecs = vecs.real[:, idx[i:]]
        if vec_func:
            vecs = vec_func(vecs)

        for i in range(vecs.shape[1]):
            vecs[:, i] /= np.linalg.norm(vecs[:, i])

        print('Perfoming mapping')
        maps = ()
        min_idx = 0
        for dx in self.row:
            map = vecs[min_idx:min_idx + dx, :self.output_dim]
            maps += (map,)
            min_idx += dx

        return maps

    def compute_distances(self):
        """Helper function to compute distances for each dataset"""
        print('Shape of Raw data')
        for i in range(self.dataset_num):
            print('Dataset {}:'.format(i), np.shape(self.dataset[i]))
            self.dataset[i] = (
                (self.dataset[i] - np.min(self.dataset[i]))
                / (np.max(self.dataset[i]) - np.min(self.dataset[i]))
            )

            if self.distance_mode == 'geodesic':
                def distance_function(df):
                    distances = geodesic_distances(df, self.kmax)
                    return np.array(distances)
            elif self.distance_mode == 'spearman':
                # Note: Method may not work if empty features are given
                def distance_function(df):
                    if df.shape[0] == 1:
                        distances = np.array([0])
                    else:
                        distances, _ = stats.spearmanr(df, axis=1)
                        if np.isnan(distances).any():
                            raise Exception(
                                'Data is not well conditioned for spearman method '
                                '(scipy.stats.spearmanr returned ``np.nan``)'
                            )
                    if len(distances.shape) == 0:
                        distances = np.array([[1, distances], [distances, 1]])
                    return (1 - np.array(distances)) / 2
            elif self.distance_mode == 'pearson':
                # Note: Method may not work if empty features are given
                # ASDDDF: Compatibility with dense matrices
                def distance_function(df):
                    if df.shape[0] == 1:
                        return np.array([0])
                    distances = np.corrcoef(df.toarray())
                    if len(distances.shape) == 0:
                        distances = np.array([[1, distances], [distances, 1]])
                    return (1 - np.array(distances)) / 2
            else:
                def distance_function(df):
                    return pairwise_distances(df, metric=self.distance_mode)

            distances = distance_function(self.dataset[i])
            self.dist.append(distances)
