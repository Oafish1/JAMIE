from itertools import product
import warnings

import anndata as ad
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn as nn  # noqa
import torch.optim as optim
import unioncom.UnionCom as uc
from unioncom.utils import (
    geodesic_distances,
    init_random_seed,
    joint_probabilities,
)

from .model import edModel
from .nn_funcs import gw_loss, knn_dist, knn_sim, nlma_loss, uc_loss  # noqa
from .utilities import time_logger, transfer_accuracy, uc_visualize


class ComManDo(uc.UnionCom):
    """
    Adaptation of https://github.com/caokai1073/UnionCom by caokai1073

    P: Correspondence prior matrix
    PF_Ratio: Ratio of priors:assumed correspondence; .5 is equal
    in_place: Whether to do the calculation in place.  Will save memory but may
        alter original data
    """
    def __init__(
        self,
        P=None,
        PF_Ratio=1,
        in_place=False,
        **kwargs
    ):
        self.P = P
        self.PF_Ratio = PF_Ratio
        self.in_place = in_place

        if 'project_mode' not in kwargs:
            kwargs['project_mode'] = 'nlma'
        # if 'distance_mode' not in kwargs:
        #     kwargs['distance_mode'] = 'spearman'

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

        # Make a copy
        if not self.in_place:
            self.dataset = self.dataset * 1

        self.dataset_num = len(self.dataset)
        for i in range(self.dataset_num):
            self.row.append(np.shape(self.dataset[i])[0])
            self.col.append(np.shape(self.dataset[i])[1])

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
                    # mat = np.eye(self.row[i])
                    mat = None
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

    def project_nlma(self, W):
        """Perform NLMA using TSNE-like backend"""
        print('-' * 33)
        print('Performing NLMA')
        assert len(W) == 2, 'Currently only compatible with 2 modalities.'

        # Default vars
        if self.P is None:
            # If not given, assume total alignment iff datasets are same size
            if self.row[0] == self.row[1]:
                self.P = np.eye(self.row[0])
            else:
                self.P = np.zeros((self.row[0], self.row[1]))

        if (self.P - np.eye(self.row[0])).sum() == 0:
            self.perfect_alignment = True
        else:
            self.perfect_alignment = False

        self.P = torch.Tensor(self.P).float().to(self.device)
        self.F = W[0][1]

        # Tuning
        # self.epoch_DNN = 500
        # self.batch_size = 6
        # self.batch_size = 177
        # self.lr = .01

        timer = time_logger()
        self.model = edModel(self.col, self.output_dim).to(self.device)
        optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

        def sim_dist_func(a, b):
            # Cosine Similarity
            sim = (
                torch.mm(a, torch.t(b))
                / (a.norm(dim=1).reshape(-1, 1) * b.norm(dim=1).reshape(1, -1))
            )
            return sim, 1-sim

            # Euclidean Distance (naïve)
            # dist = torch.zeros(a.size()[0], b.size()[0])
            # for i in range(a.size()[0]):
            #     for j in range(b.size()[0]):
            #         dist[i][j] = torch.linalg.norm(a[i] - b[j])
            # sim = 1 / (1+dist)
            # return sim, dist

        self.model.train()

        # Convert data
        for i in range(self.dataset_num):
            self.dataset[i] = torch.from_numpy(self.dataset[i]).float().to(self.device)

        # Batch size setup, mainly from UnionCom
        len_dataloader = np.int(np.max(self.row)/self.batch_size)
        if len_dataloader == 0:
            len_dataloader = 1
            self.batch_size = np.max(self.row)
        timer.log('Setup')

        for epoch in range(self.epoch_DNN):
            epoch_loss = 0
            for batch_idx in range(len_dataloader):
                batch_loss = 0

                # Random samples
                if self.perfect_alignment:
                    set_rand = np.random.choice(range(self.row[i]), self.batch_size, replace=False)
                random_batch = [
                    set_rand if self.perfect_alignment else
                    np.random.choice(range(self.row[i]), self.batch_size, replace=False)
                    # np.random.choice(range(self.row[i]), self.batch_size - i, replace=False)
                    for i in range(self.dataset_num)
                ]
                data = [self.dataset[i][random_batch[i]] for i in range(self.dataset_num)]

                # P setup
                P = self.P[random_batch[0]][:, random_batch[1]]
                if P.sum() != 0:
                    P /= P.sum()

                # F setup
                F = self.F[random_batch[0]][:, random_batch[1]]

                # KEY PART FOR ACC
                # Use KNN, needs to be adapted to diff num samples
                F = knn_dist(F) if self.perfect_alignment else knn_sim(F)
                F = torch.from_numpy(F).float().to(self.device)

                F_inv = 1 - F
                F /= F.sum()
                F_inv /= F_inv.sum()
                timer.log('Get subset samples')

                # Aggregate correspondence
                corr = self.PF_Ratio * P + (1-self.PF_Ratio) * F

                # Run model
                embedded, reconstructed = self.model(*data, corr=corr)
                timer.log('Run model')

                # Reconstruction error
                reconstruction_diff = sum(
                    (reconstructed[i] - data[i]).square().sum()
                    for i in range(self.dataset_num)
                )
                reconstruction_loss = 1e-3 * reconstruction_diff
                batch_loss += reconstruction_loss
                timer.log('Reconstruction loss')

                # Difference
                csim, cdiff = sim_dist_func(embedded[0], embedded[1])

                # Alignment loss
                alignment_loss = 2e-4 * cdiff[P > 0].sum()
                batch_loss += alignment_loss
                timer.log('Aligned loss')

                # Cross loss using F
                weighted_F_cdiff = cdiff * F
                cross_loss = 1e+1 * weighted_F_cdiff.sum()
                batch_loss += cross_loss
                timer.log('F-cross loss')

                # Inverse cross loss using F
                weighted_F_inv_csim = csim * F_inv
                inv_cross_loss = 3e+0 * weighted_F_inv_csim.sum()
                batch_loss += inv_cross_loss
                timer.log('F-inv-cross loss')

                # Debug print losses
                # if epoch == self.epoch_DNN - 1:
                #     print()
                #     print(reconstruction_loss)
                #     print(alignment_loss)
                #     print(cross_loss)
                #     print(inv_cross_loss)
                #     print()

                # Record loss
                epoch_loss += batch_loss / len_dataloader

                # Step
                loss = batch_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                timer.log('Step')

            if (epoch+1) % self.log_DNN == 0:
                print(f'epoch:[{epoch+1:d}/{self.epoch_DNN}]: loss:{epoch_loss.data.item():4f}')
                # self.Visualize(self.dataset, [p.detach().cpu().numpy() for p in primes])

        self.model.eval()
        corr_P = self.P / self.P.sum()
        corr_F = torch.from_numpy(knn_sim(self.F)).float().to(self.device)
        corr_F /= corr_F.sum()
        corr = self.PF_Ratio * corr_P + (1-self.PF_Ratio) * corr_F
        integrated_data, _ = self.model(*self.dataset, corr=corr)
        integrated_data = [d.detach().cpu().numpy() for d in integrated_data]
        timer.log('Output')
        print("Finished Mapping!")
        timer.aggregate()
        return integrated_data

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

            self.distance_function = distance_function
            distances = self.distance_function(self.dataset[i])
            self.dist.append(distances)

    def test_closer(
        self,
        integrated_data,
        distance_metric=lambda x: pairwise_distances(x, metric='euclidean'),
    ):
        """Test fraction of samples closer than the true match"""
        # ASDF: 3+ datasets and non-aligned data
        assert len(integrated_data) == 2, 'Two datasets are supported for FOSCTTM'

        if distance_metric is None:
            distance_metric = self.distance_function
        distances = distance_metric(np.concatenate(integrated_data, axis=0))
        size = integrated_data[0].shape[0]
        raw_count_closer = 0
        for i in range(size):
            # A -> B
            local_dist = distances[i][size:]
            raw_count_closer += np.sum(local_dist < local_dist[i])
            # B -> A
            local_dist = distances[size+i][:size]
            raw_count_closer += np.sum(local_dist < local_dist[i])
        foscttm = raw_count_closer / (2 * size**2)
        print(f'foscttm: {foscttm}')
        return foscttm

    def test_label_dist(
        self,
        integrated_data,
        datatype,
        distance_metric=lambda x: pairwise_distances(x, metric='euclidean'),
    ):
        """Test average distance by label"""
        # ASDF: 3+ datasets
        assert len(integrated_data) == 2, 'Two datasets are supported for ``label_dist``'

        if distance_metric is None:
            distance_metric = self.distance_function
        data = np.concatenate(integrated_data, axis=0)
        labels = np.concatenate(datatype)

        # Will double-count aligned sample
        average_representation = {}
        for label in np.unique(labels):
            average_representation[label] = np.average(data[labels == label, :], axis=0)
        dist = distance_metric(np.array(list(average_representation.values())))
        print(f'Inter-label distances ({list(average_representation.keys())}):')
        print(dist)
        return average_representation.keys(), dist

    def test_LabelTA(self, integrated_data, datatype):
        """Modified version of UnionCom ``test_LabelTA`` to return acc"""
        acc = transfer_accuracy(integrated_data[0], integrated_data[1], datatype[0], datatype[1])
        print(f"label transfer accuracy: {acc}")
        return acc

    def Visualize(self, data, integrated_data, datatype=None, mode=None):
        """In-class API for modified visualization function"""
        uc_visualize(data, integrated_data, datatype=datatype, mode=mode)
