from itertools import product
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import torch
import unioncom.UnionCom as uc
from unioncom.utils import geodesic_distances, init_random_seed, joint_probabilities

from .maninetcluster.neighborhood import laplacian


class NMAcom(uc.UnionCom):
    """Modification of https://github.com/caokai1073/UnionCom by caokai1073"""
    def __init__(self, **kwargs):
        if 'project_mode' not in kwargs:
            kwargs['project_mode'] = 'nlma'
        super().__init__(**kwargs)

    def fit_transform(self, dataset=None):
        """The only addition here is the `'nlma'` option for `project_mode`"""
        distance_modes = [
            'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis',
            'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming',
            'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean',
            'haversine',
        ]

        if self.integration_type not in ['BatchCorrect', 'MultiOmics']:
            raise Exception('integration_type error! Enter MultiOmics or BatchCorrect.')
        if self.distance_mode != 'geodesic' and self.distance_mode not in distance_modes:
            raise Exception('distance_mode error! Enter a correct distance_mode.')
        if self.project_mode not in ('tsne', 'barycentric', 'nlma'):
            raise Exception("Choose correct project_mode: 'tsne, barycentric, or nlma'")

        time1 = time.time()
        init_random_seed(self.manual_seed)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataset_num = len(dataset)
        for i in range(dataset_num):
            self.row.append(np.shape(dataset[i])[0])
            self.col.append(np.shape(dataset[i])[1])

        # Compute the distance matrix
        print('Shape of Raw data')
        for i in range(dataset_num):
            print('Dataset {}:'.format(i), np.shape(dataset[i]))
            dataset[i] = (
                (dataset[i] - np.min(dataset[i]))
                / (np.max(dataset[i]) - np.min(dataset[i]))
            )

            if self.distance_mode == 'geodesic':
                distances = geodesic_distances(dataset[i], self.kmax)
                self.dist.append(np.array(distances))
            else:
                distances = pairwise_distances(dataset[i], metric=self.distance_mode)
                self.dist.append(distances)

            if self.integration_type == 'BatchCorrect':
                if self.distance_mode not in distance_modes:
                    raise Exception('Note that BatchCorrect needs aligned features.')
                else:
                    if self.col[i] != self.col[-1]:
                        raise Exception('BatchCorrect needs aligned features.')
                    cor_distances = pairwise_distances(
                        dataset[i],
                        dataset[-1],
                        metric=self.distance_mode,
                    )
                    self.cor_dist.append(cor_distances)

        # Find correspondence between samples
        match_result = self.match(dataset=dataset)

        #  Project to common embedding
        if self.project_mode == 'tsne':
            pairs_x = []
            pairs_y = []
            for i in range(dataset_num-1):
                cost = np.max(match_result[i])-match_result[i]
                # NMAcom modification for priors
                row_ind, col_ind = linear_sum_assignment(cost)
                pairs_x.append(row_ind)
                pairs_y.append(col_ind)

            P_joint = []
            time1 = time.time()
            for i in range(dataset_num):
                P_joint.append(joint_probabilities(self.dist[i], self.perplexity))
            for i in range(dataset_num):
                if self.col[i] > 50:
                    dataset[i] = PCA(n_components=50).fit_transform(dataset[i])
                    self.col[i] = 50
            integrated_data = self.project_tsne(dataset, pairs_x, pairs_y, P_joint)
        elif self.project_mode == 'barycentric':
            integrated_data = self.project_barycentric(dataset, match_result)
        elif self.project_mode == 'nlma':
            integrated_data = self.project_nlma(dataset, match_result)

        print('---------------------------------')
        print('NMAcom Done!')
        time2 = time.time()
        print('time:', time2-time1, 'seconds')

        return integrated_data

    def match(self, dataset):
        """Find correspondence between multi-omics datasets"""

        dataset_num = len(dataset)

        if self.project_mode == 'nlma':
            cor_pairs = dataset_num * [dataset_num * [None]]
            for i in range(dataset_num):
                for j in range(i, dataset_num):
                    print("---------------------------------")
                    print(f'Find correspondence between Dataset {i + 1} and Dataset {j + 1}')
                    F = self.Prime_Dual([self.dist[i], self.dist[i]],
                                        dx=self.col[i],
                                        dy=self.col[j])
                    cor_pairs[i][j] = F

                    # Save some computation
                    if i != j:
                        cor_pairs[j][i] = F.T
        else:
            cor_pairs = []
            for i in range(dataset_num-1):
                print("---------------------------------")
                print(f'Find correspondence between Dataset {i + 1} and Dataset {len(dataset)}')
                if self.integration_type == "MultiOmics":
                    cor_pairs.append(self.Prime_Dual([self.dist[i], self.dist[-1]],
                                                     dx=self.col[i],
                                                     dy=self.col[-1]))
                else:
                    cor_pairs.append(self.Prime_Dual(self.cor_dist[i]))

        print("Finished Matching!")
        return cor_pairs

    def project_nlma(self, dataset, F_list):
        """
        Projects using `F` matrices as correspondence using NLMA, heavily
        relies on methodology and code from
        https://github.com/daifengwanglab/ManiNetCluster
        """
        assert len(dataset) == 2, 'NLMA only supports 2 datasets'

        mu = .9
        eps = 1e-8

        # Set up manifold
        dim = len(F_list)
        W = F_list

        # TODO: Verify coef structure for >2 modalities
        coef = (1-mu) * np.ones((dim, dim))
        np.fill_diagonal(coef, mu)
        for i, j in product(*(2 * [range(dim)])):
            W[i][j] *= coef[i, j]
        W = np.asarray(np.bmat(W))
        L = laplacian(W)

        # Perform decomposition
        vec_func = None

        vals, vecs = np.linalg.eig(L)
        idx = np.argsort(vals)
        for i in range(len(idx)):
            if vals[idx[i]] >= eps:
                break
        vecs = vecs.real[:, idx[i:]]
        if vec_func:
            vecs = vec_func(vecs)

        for i in range(vecs.shape[1]):
            vecs[:, i] /= np.linalg.norm(vecs[:, i])

        maps = []
        min_idx = 0
        for data in dataset:
            dx = data.shape[0]
            map = vecs[min_idx:min_idx + dx, :self.output_dim]
            maps.append(map)
            min_idx += dx

        return tuple(maps)
