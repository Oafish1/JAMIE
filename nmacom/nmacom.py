import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import torch
import unioncom.UnionCom as uc
from unioncom.utils import geodesic_distances, init_random_seed, joint_probabilities

from .maninetcluster import manifold_nonlinear
from .maninetcluster.correspondence import Correspondence
from .maninetcluster.neighborhood import neighbor_graph


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
        pairs_x = []
        pairs_y = []
        match_result = self.match(dataset=dataset)
        for i in range(dataset_num-1):
            cost = np.max(match_result[i])-match_result[i]
            # NMAcom modification for priors
            row_ind, col_ind = linear_sum_assignment(cost)
            pairs_x.append(row_ind)
            pairs_y.append(col_ind)

        #  Project to common embedding
        if self.project_mode == 'tsne':
            # NMAcom modification for NLMA
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
        else:
            raise Exception("Choose correct project_mode: 'tsne, barycentric, or nlma'")

        print('---------------------------------')
        print('unionCom Done!')
        time2 = time.time()
        print('time:', time2-time1, 'seconds')

        return integrated_data

    def project_nlma(self, dataset, F):
        """Projects using the `F` matrix as correspondence using NLMA"""
        assert len(dataset) == 2, 'NLMA only supports 2 datasets'

        corr = Correspondence(matrix=np.array(F[0]))
        num_dims = 10
        W = [neighbor_graph(data, k=10) for data in dataset]
        # X,Y,corr,num_dims,Wx,Wy,mu=0.9,eps=1e-8
        return manifold_nonlinear(*dataset, corr, num_dims, *W)
