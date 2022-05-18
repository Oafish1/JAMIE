from itertools import product
from math import prod
import warnings

import anndata as ad
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
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
from .utilities import time_logger, uc_visualize


class ComManDo(uc.UnionCom):
    """
    Adaptation of https://github.com/caokai1073/UnionCom by caokai1073

    P: Correspondence prior matrix
    PF_Ratio: Ratio of priors:assumed correspondence; .5 is equal, 1 is only P
    in_place: Whether to do the calculation in place.  Will save memory but may
        alter original data
    """
    def __init__(
        self,
        P=None,
        match_result=None,
        PF_Ratio=1,
        corr_method='unioncom',
        dist_method='cosine',
        in_place=False,
        loss_weights=None,
        model_class=edModel,
        pca_dim=None,
        use_early_stop=True,
        min_increment=.1,
        max_steps_without_increment=200,
        debug=False,
        **kwargs
    ):
        self.P = P
        self.match_result = match_result
        self.PF_Ratio = PF_Ratio
        self.corr_method = corr_method
        self.dist_method = dist_method
        self.in_place = in_place
        self.loss_weights = loss_weights
        self.model_class = model_class
        self.pca_dim = pca_dim

        self.use_early_stop = use_early_stop
        self.min_increment = min_increment
        self.max_steps_without_increment = max_steps_without_increment

        self.debug = debug

        # Default changes
        # if 'distance_mode' not in kwargs:
        #     kwargs['distance_mode'] = 'spearman'
        if 'project_mode' not in kwargs:
            kwargs['project_mode'] = 'commando'
        if 'log_pd' not in kwargs:
            kwargs['log_pd'] = 500

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
        if self.project_mode not in ('commando', 'tsne'):
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
        self.col = []
        self.row = []
        for i in range(self.dataset_num):
            self.row.append(np.shape(self.dataset[i])[0])
            self.col.append(np.shape(self.dataset[i])[1])

        # Compute the distance matrix
        self.compute_distances()
        time.log('Distance')

        # Find correspondence between samples
        self.match_result = self.match() if self.match_result is None else self.match_result
        pairs_x = []
        pairs_y = []
        for i in range(self.dataset_num - 1):
            cost = np.max(self.match_result[i]) - self.match_result[i]
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
        elif self.project_mode == 'commando':
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
                    mat = self.match_result[k]
                    k += 1
                match_matrix[i][j] = mat
            integrated_data = self.project_commando(match_matrix)
        time.log('Mapping')

        print('-' * 33)
        print('ComManDo Done!')
        time.aggregate()
        print()

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
                if self.corr_method == 'unioncom':
                    F = self.Prime_Dual(
                        [self.dist[i], self.dist[j]],
                        dx=self.col[i],
                        dy=self.col[j],
                    )
                elif self.corr_method == 'commando':
                    F = self.com_corr([self.dist[i], self.dist[j]])
                cor_pairs.append(F)

        print("Finished Matching!")
        return cor_pairs

    def com_corr(self, dist):
        """Estimate correspondence"""
        Kx, Ky = dist
        n, m = np.shape(Kx)[0], np.shape(Ky)[0]
        Kx = torch.Tensor(Kx)
        Ky = torch.Tensor(Ky)

        # Params
        dim = 20
        keep_prob = .35
        epochs = 10001
        epoch_p = 2000

        # Initialize
        a = torch.rand(1, requires_grad=True)
        F = torch.rand(dim, dim, requires_grad=True)
        Tx = torch.rand(dim, n, requires_grad=True)
        Ty = torch.rand(dim, m, requires_grad=True)

        # Step
        print('Clustering')
        optimizer = optim.RMSprop([Tx, Ty], lr=.01)
        for i in range(epochs):
            # loss = (a*Kx - torch.mm(F, torch.mm(Ky, F.T))).square().sum()
            maskx = torch.diag(1.*(torch.rand(n) > (1-keep_prob)))
            masky = torch.diag(1.*(torch.rand(m) > (1-keep_prob)))
            tx = torch.mm(Tx, maskx)
            ty = torch.mm(Ty, masky)
            loss = (
                torch.mm(tx, torch.mm(Kx, tx.T))
                - torch.mm(ty, torch.mm(Ky, ty.T))
            ).square().sum()

            if (i % epoch_p) == 0:
                print(f'loss: {float(loss.cpu().detach())}')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Casting')
        Tx = Tx.detach()
        Ty = Ty.detach()
        optimizer = optim.RMSprop([a, F], lr=.1)
        for i in range(epochs):
            Fc = torch.mm(Tx.T, torch.mm(F, Ty))
            loss = (a*Kx - torch.mm(Fc, torch.mm(Ky, Fc.T))).square().sum()

            if (i % epoch_p) == 0:
                print(f'loss: {float(loss.cpu().detach())}')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        corr = torch.mm(Tx.T, torch.mm(F, Ty)).cpu().detach()
        k = 5
        corr_idx = torch.argsort(corr, dim=1, descending=True)[:, :k]
        corr = torch.zeros(n, m).cpu()
        corr[corr_idx] = 1
        return corr.numpy()

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

    def project_commando(self, W):
        """Perform alignment using TSNE-like backend"""
        print('-' * 33)
        print('Train coupled autoencoders')
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
        self.F = torch.Tensor(W[0][1]).float().to(self.device)

        # Debugging
        # self.epoch_DNN = 500
        # self.lr = .1
        # self.batch_size = 8
        # self.batch_size = 177

        timer = time_logger()
        if self.pca_dim is not None:
            pca_list = []
            for dim, data in zip(self.pca_dim, self.dataset):
                if dim is not None:
                    pca = PCA(n_components=dim).fit(data)
                    pca_list.append(pca.transform)
                else:
                    pca_list.append(lambda x: x)
            # Python bug?  Doesn't work, overwrites pca
            # pca_list = [lambda x: pca.transform(x) for pca in pca_list]

            # Transform datasets (Maybe find less destructive way?)
            self.dataset = [pca_transform(x) for pca_transform, x in zip(pca_list, self.dataset)]
            self.col = [x.shape[1] for x in self.dataset]
        else:
            pca_list = None
        self.model = (
            self.model_class(
                self.col,
                self.output_dim,
                preprocessing=pca_list,
            ).to(self.device))
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        def sim_diff_func(a, b):
            if self.dist_method == 'cosine':
                # Cosine Similarity
                sim = (
                    torch.mm(a, torch.t(b))
                    / (a.norm(dim=1).reshape(-1, 1) * b.norm(dim=1).reshape(1, -1))
                )
                # return sim, 1-sim
                diff = 1 - sim
                sim[sim < 0] = 0
                diff[diff < 0] = 0
                return sim, diff

            elif self.dist_method == 'euclidean':
                # Euclidean Distance
                dist = torch.cdist(a, b, p=2)
                dist = dist / dist.max()
                sim = 1 / (1+dist)
                sim = 2 * sim - 1  # This one scaling line makes the entire algorithm work
                sim = sim / sim.max()
                return sim, dist

        self.model.train()

        # Convert data
        for i in range(self.dataset_num):
            self.dataset[i] = torch.from_numpy(self.dataset[i]).float().to(self.device)

        # Batch size setup, mainly from UnionCom
        len_dataloader = np.int(np.max(self.row)/self.batch_size)
        if len_dataloader == 0:
            len_dataloader = 1
            self.batch_size = np.max(self.row)

        # Early stopping setup
        best_loss = np.inf
        streak = 0

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
                if P.absolute().max() != 0:
                    P /= P.absolute().max()

                # F setup
                F = self.F[random_batch[0]][:, random_batch[1]]
                if F.absolute().max() != 0:
                    F /= F.absolute().max()
                F_inv = 1 - F
                if F_inv.absolute().max() != 0:
                    F_inv /= F_inv.absolute().max()

                timer.log('Get subset samples')

                # Aggregate correspondence
                corr = self.PF_Ratio * P + (1-self.PF_Ratio) * F

                # Run model
                embedded, reconstructed = self.model(*data, corr=corr)
                timer.log('Run model')

                # Loss bookkeeping
                losses = []

                # Reconstruction error
                reconstruction_diff = sum(
                    (reconstructed[i] - data[i]).square().sum() / prod(data[i].shape)
                    for i in range(self.dataset_num)
                ) / self.dataset_num
                reconstruction_loss = reconstruction_diff
                losses.append(reconstruction_loss)
                timer.log('Reconstruction loss')

                # Difference
                csim, cdiff = sim_diff_func(embedded[0], embedded[1])
                csim0, cdiff0 = sim_diff_func(embedded[0], embedded[0])
                csim1, cdiff1 = sim_diff_func(embedded[1], embedded[1])
                timer.log('Difference calculation')

                # Alignment loss
                if P.absolute().sum() != 0:
                    weighted_P_cdiff = cdiff * P
                    alignment_loss = weighted_P_cdiff.absolute().sum() / P.absolute().sum()
                    losses.append(alignment_loss)
                    timer.log('Aligned loss')

                # Cross loss using F
                weighted_F_cdiff = cdiff * F
                cross_loss = weighted_F_cdiff.absolute().sum() / F.absolute().sum()
                losses.append(cross_loss)
                timer.log('F-cross loss')

                # Inverse cross loss using F
                weighted_F_inv_csim = csim * F_inv
                inv_cross_loss = weighted_F_inv_csim.absolute().sum() / F_inv.absolute().sum()
                losses.append(inv_cross_loss)
                timer.log('F-inv-cross loss')

                # # Distance loss
                # _, dist_diff0 = sim_dist_func(data[0], data[0])
                # _, dist_diff1 = sim_dist_func(data[1], data[1])
                # dist_loss0 = (cdiff0 - dist_diff0).square().sum() / prod(cdiff0.shape)
                # dist_loss1 = (cdiff1 - dist_diff1).square().sum() / prod(cdiff1.shape)
                # dist_loss = (dist_loss0 + dist_loss1) / 2
                # losses.append(dist_loss)
                # timer.log('Distance Loss')

                # # Distance F loss
                # _, dist_diff0 = sim_dist_func(data[0], data[0])
                # _, dist_diff1 = sim_dist_func(data[1], data[1])
                # dist_loss0 = (
                #     torch.mm(torch.linalg.pinv(F), cdiff0)
                #     - torch.mm(torch.linalg.pinv(F), dist_diff0)
                # ).square().sum() / prod(cdiff0.shape)
                # dist_loss1 = (
                #     torch.mm(cdiff1, torch.t(F)) - torch.mm(dist_diff1, torch.t(F))
                # ).square().sum() / prod(cdiff1.shape)
                # dist_loss = (dist_loss0 + dist_loss1) / 2
                # losses.append(dist_loss)
                # timer.log('Distance F Loss')

                # # Com loss
                # com_diff = cdiff0 - torch.mm(F, torch.mm(cdiff1, torch.t(F)))
                # com_loss = (com_diff / (1+cdiff0)).square().sum()
                # losses.append(com_loss)
                # timer.log('Com Loss')

                # # Repulsion loss (Promising, almost equiv to inv-cross)
                # repulsion_loss = (
                #     # (csim.sum() + csim0.sum() + csim1.sum()) /
                #     # (prod(csim.shape) + prod(csim0.shape) + prod(csim1.shape))
                #     (csim0.sum() + csim1.sum()) / (prod(csim0.shape) + prod(csim1.shape))
                # )
                # losses.append(repulsion_loss)
                # timer.log('Repulsion loss')

                # # Efficiency loss (Promising)
                # total_correlation = 0
                # for i in range(self.dataset_num):
                #     adjusted = embedded[i] - embedded[i].mean(axis=0, keepdim=True)
                #     mat = torch.mm(torch.t(adjusted), adjusted)
                #     for j in range(self.output_dim):
                #         for k in range(j+1, self.output_dim):
                #             correlation = mat[j, k] / (mat[j, j] * mat[k, k]).sqrt()
                #             total_correlation += correlation.square()
                # efficiency_loss = total_correlation / (self.output_dim**2 - self.output_dim)
                # losses.append(efficiency_loss)
                # timer.log('Collapse loss')

                # # Magnitude loss
                # max0 = embedded[0].abs().mean(axis=1).max()
                # max1 = embedded[1].abs().mean(axis=1).max()
                # mag_loss = 1e-1 * (max0 + max1)
                # losses.append(mag_loss)
                # timer.log('Magnitude loss')

                # # Zero loss
                # min0 = 1 / embedded[0].abs()
                # min1 = 1 / embedded[1].abs()
                # zero_loss = (min0.sum() + min1.sum()) / (2 * prod(embedded[0].shape))
                # losses.append(zero_loss)
                # timer.log('Collapse loss')

                # # Axis stick loss
                # min0 = 1 / embedded[0].mean(axis=1).abs()
                # min1 = 1 / embedded[1].mean(axis=1).abs()
                # as_loss = 1e-4 * (min0.sum() + min1.sum())
                # losses.append(as_loss)
                # timer.log('Collapse loss')

                # # Collapse loss
                # min0 = 1 / embedded[0].abs().mean(axis=1).min()
                # min1 = 1 / embedded[1].abs().mean(axis=1).min()
                # collapse_loss = 1e-1 * (min0 + min1)
                # losses.append(collapse_loss)
                # timer.log('Collapse loss')

                # # F loss
                # F1 = torch.mm(torch.t(F), torch.mm(cdiff0, F)) - cdiff1
                # F1 = F1.square().sum()
                # F2 = torch.mm(F, torch.mm(cdiff1, torch.t(F))) - cdiff0
                # F2 = F2.square().sum()
                # F_loss = 3e-3 * (F1 + F2)
                # losses.append(F_loss)
                # timer.log('F loss')

                # Record loss
                if self.loss_weights is not None:
                    batch_loss = sum([lo * wt for lo, wt in zip(losses, self.loss_weights)])
                else:
                    batch_loss = sum(losses)
                epoch_loss += batch_loss / len_dataloader

                # Append losses
                batch_loss.backward()

            # Step
            optimizer.step()
            optimizer.zero_grad()
            timer.log('Step')

            # CLI Printing
            if (epoch+1) % self.log_DNN == 0:
                print(f'epoch:[{epoch+1:d}/{self.epoch_DNN}]: loss:{epoch_loss.data.item():4f}')

            # Debug Printing
            if (epoch+1) % 100 == 0 and self.debug:
                if self.loss_weights is not None:
                    print([lo.detach().cpu() * wt for lo, wt in zip(losses, self.loss_weights)])
                else:
                    print([lo.detach().cpu() for lo in losses])

            # Early stopping
            epsilon = best_loss - epoch_loss
            if epsilon > self.min_increment:
                best_loss = epoch_loss
                streak = 0
            else:
                streak += 1
            if streak >= self.max_steps_without_increment and self.use_early_stop:
                break

        self.model.eval()
        corr_P = self.P / self.P.sum()
        corr_F = self.F
        # corr_F = knn_dist(corr_F) if self.perfect_alignment else knn_sim(corr_F)
        corr_F /= corr_F.absolute().max()
        corr = self.PF_Ratio * corr_P + (1-self.PF_Ratio) * corr_F
        integrated_data, _ = self.model(*self.dataset, corr=corr)
        integrated_data = [d.detach().cpu().numpy() for d in integrated_data]
        timer.log('Output')
        print("Finished Mapping!")
        # timer.aggregate()
        return integrated_data

    def modal_predict(self, data, modality):
        """Predict the opposite modality from dataset ``data`` in modality ``modality``"""
        assert self.model is not None, 'Model must be trained before modal prediction.'

        to_modality = (modality + 1) % self.dataset_num
        pre_function = self.model.preprocessing[modality]
        return self.model.decoders[to_modality](
            self.model.encoders[modality](
                torch.tensor(pre_function(data)).float()
            )).detach().cpu().numpy()

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
        verbose=True,
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
        if verbose:
            print(f'Inter-label distances ({list(average_representation.keys())}):')
            print(dist)
        return np.array(list(average_representation.keys())), dist

    def test_LabelTA(self, integrated_data, datatype, k=None, return_k=False):
        """Modified version of UnionCom ``test_LabelTA`` to return acc"""
        if k == None:
            # Set to 20% of avg class size if no k provided
            total_size = min(*[len(d) for d in datatype])
            num_classes = len(np.unique(np.concatenate(datatype)).flatten())
            k = int(.2 * total_size / num_classes)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(integrated_data[1], datatype[1])
        type1_predict = knn.predict(integrated_data[0])
        count = 0
        for label1, label2 in zip(type1_predict, datatype[0]):
        	if label1 == label2:
        		count += 1
        acc = count / len(datatype[0])
        print(f"label transfer accuracy: {acc}")
        if return_k:
            return acc, k
        return acc

    def Visualize(self, data, integrated_data, datatype=None, mode=None):
        """In-class API for modified visualization function"""
        uc_visualize(data, integrated_data, datatype=datatype, mode=mode)
