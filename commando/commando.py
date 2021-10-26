from itertools import product
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import torch
import unioncom.UnionCom as uc
from unioncom.utils import geodesic_distances, init_random_seed, joint_probabilities

from .maninetcluster.neighborhood import laplacian
from .utilities import time_logger


class ComManDo(uc.UnionCom):
    """Adaptation of https://github.com/caokai1073/UnionCom by caokai1073"""
    def __init__(
        self,
        gradient_reduction=None,
        gradient_reduction_threshold=.99,
        prime_dual_verbose=False,
        two_step_aggregation=None,
        two_step_num_partitions=None,
        two_step_pd_large=None,
        **kwargs,
    ):
        if 'project_mode' not in kwargs:
            kwargs['project_mode'] = 'nlma'

        self.gradient_reduction = gradient_reduction
        self.gradient_reduction_threshold = gradient_reduction_threshold

        self.two_step_pd_large = two_step_pd_large
        self.two_step_aggregation = two_step_aggregation
        self.two_step_num_partitions = two_step_num_partitions

        self.prime_dual_verbose = prime_dual_verbose

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
            raise Exception("Choose correct project_mode: 'tsne, barycentric, or nlma'.")
        if self.project_mode == 'nlma' and self.integration_type == 'BatchCorrect':
            raise Exception(
                "project_mode: 'nlma' is incompatible with integration_type: 'BatchCorrect'."
            )
        if self.gradient_reduction is not None and self.project_mode != 'MultiOmics':
            raise Exception('gradient_reduction cannot be used with project_mode: MultiOmics.')

        time1 = time.time()
        init_random_seed(self.manual_seed)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataset_num = len(dataset)
        for i in range(dataset_num):
            self.row.append(np.shape(dataset[i])[0])
            self.col.append(np.shape(dataset[i])[1])
        if self.project_mode == 'nlma' and not (np.array(self.row) == self.row[0]).all():
            raise Exception("project_mode: 'nlma' requres aligned datasets.")

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
                distances = np.array(distances)
            else:
                distances = pairwise_distances(dataset[i], metric=self.distance_mode)
            self.dist.append(torch.from_numpy(distances).float().to(self.device))

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
                # TODO: modification for priors
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

        print('-' * 33)
        print('ComManDo Done!')
        time2 = time.time()
        print('Time:', time2-time1, 'seconds')

        return integrated_data

    def match(self, dataset):
        """Find correspondence between multi-omics datasets"""
        print('Device:', self.device)
        dataset_num = len(dataset)

        if self.project_mode == 'nlma':
            cor_pairs = dataset_num * [dataset_num * [None]]
            for i in range(dataset_num):
                for j in range(i, dataset_num):
                    print('-' * 33)
                    print(f'Find correspondence between Dataset {i + 1} and Dataset {j + 1}')
                    if self.two_step_num_partitions is not None:
                        # First step
                        # Create groupings
                        idx_all = torch.arange(self.row[i])
                        # TODO: Work on all datasets
                        if self.two_step_aggregation == 'agglomerative':
                            agg_groups = (
                                AgglomerativeClustering(
                                    n_clusters=self.two_step_num_partitions,
                                    affinity='precomputed',
                                    linkage='average',
                                )
                                .fit(self.dist[i])
                                .labels_
                            )
                            idx_groups = [
                                idx_all[agg_groups == i]
                                for i in range(self.two_step_num_partitions)
                            ]
                        else:
                            np.random.shuffle(idx_all)
                            idx_groups = np.array_split(idx_all, self.two_step_num_partitions)

                        F_diag = []
                        for k, idx in enumerate(idx_groups):
                            print(f'Calculating intra-group F #{k + 1}')
                            F_diag.append(self.Prime_Dual(
                                [self.dist[i][idx][:, idx], self.dist[j][idx][:, idx]],
                                dx=len(idx),
                                dy=len(idx),
                            ))

                        # Reconstruct and unshuffle large F
                        print('Constructing large F')
                        F = torch.block_diag(*F_diag)
                        idx_all_inv = np.argsort(idx_all)
                        F = F[idx_all_inv][:, idx_all_inv]

                        # Second step
                        print('Calculating inter-group F')
                        # Compute representative points (distance)
                        rep_transform = torch.zeros((self.row[i], self.two_step_num_partitions))
                        for k, idx in enumerate(idx_groups):
                            rep_transform[idx, k] = 1

                        def shrink_matrix(m):
                            return torch.mm(
                                torch.t(rep_transform),
                                torch.mm(m, rep_transform),
                            )

                        def expand_matrix(m):
                            return torch.mm(
                                rep_transform,
                                torch.mm(m, torch.t(rep_transform)),
                            )

                        i_dist = shrink_matrix(self.dist[i])
                        j_dist = shrink_matrix(self.dist[j])

                        # Calculate inter-pseudo F
                        rep_F = self.Prime_Dual(
                            [i_dist, j_dist],
                            dx=self.two_step_num_partitions,
                            dy=self.two_step_num_partitions,
                            epoch_override=self.two_step_pd_large,
                        )
                        rep_F = expand_matrix(rep_F)

                        # Add inter-pseudocell data
                        F += rep_F
                    else:
                        F = self.Prime_Dual(
                            [self.dist[i], self.dist[j]],
                            dx=self.col[i],
                            dy=self.col[j],
                        )
                    cor_pairs[i][j] = F
                    if i != j:
                        cor_pairs[j][i] = F.T

        else:
            cor_pairs = []
            for i in range(dataset_num-1):
                print('-' * 33)
                print(f'Find correspondence between Dataset {i + 1} and Dataset {len(dataset)}')
                if self.integration_type == "MultiOmics":
                    cor_pairs.append(self.Prime_Dual([self.dist[i], self.dist[-1]],
                                                     dx=self.col[i],
                                                     dy=self.col[-1]))
                else:
                    cor_pairs.append(self.Prime_Dual(self.cor_dist[i]))

        print("Finished Matching!")
        return cor_pairs

    def Prime_Dual(self, dist, dx=None, dy=None, epoch_override=None):
        """Prime dual combined with Adam algorithm to find the local optimal soluation"""
        if self.integration_type == "MultiOmics":
            Kx = dist[0]
            Ky = dist[1]
            N = np.int(np.maximum(len(Kx), len(Ky)))
            Kx = Kx / N
            Ky = Ky / N
            Kx = Kx.float().to(self.device)
            Ky = Ky.float().to(self.device)
            a = np.sqrt(dy/dx)
            m = np.shape(Kx)[0]
            n = np.shape(Ky)[0]

        else:
            m = np.shape(dist)[0]
            n = np.shape(dist)[1]
            a = 1
            dist = dist.float().to(self.device)

        # F = np.zeros((m, n))
        F = torch.zeros((m, n)).float().to(self.device)
        Im = torch.ones((m, 1)).float().to(self.device)
        In = torch.ones((n, 1)).float().to(self.device)
        Lambda = torch.zeros((n, 1)).float().to(self.device)
        Mu = torch.zeros((m, 1)).float().to(self.device)
        S = torch.zeros((n, 1)).float().to(self.device)

        pho1 = 0.9
        pho2 = 0.999
        delta = 10e-8
        Fst_moment = torch.zeros((m, n)).float().to(self.device)
        Snd_moment = torch.zeros((m, n)).float().to(self.device)

        i = 0
        timer = time_logger(record=self.prime_dual_verbose)

        epochs = self.epoch_pd
        if epoch_override is not None:
            epochs = epoch_override
        while(i < epochs):
            if self.gradient_reduction is not None:
                if self.gradient_reduction_threshold is not None:
                    # Statistical selection
                    m_reduction_map = torch.rand((self.gradient_reduction, m))
                    n_reduction_map = torch.rand((self.gradient_reduction, n))
                    m_reduction_map /= torch.reshape(torch.sum(m_reduction_map, 1),
                                                     (self.gradient_reduction, -1))
                    n_reduction_map /= torch.reshape(torch.sum(n_reduction_map, 1),
                                                     (self.gradient_reduction, -1))
                else:
                    # Binary selection
                    # Gradients are far more volatile with gradient reduction, which
                    # necessitates binary gradient reduction
                    m_reduction_map = (
                        torch.rand((self.gradient_reduction, m))
                        > self.gradient_reduction_threshold
                    )
                    n_reduction_map = (
                        torch.rand((self.gradient_reduction, n))
                        > self.gradient_reduction_threshold
                    )
                    m_reduction_map = m_reduction_map.to(torch.float32)
                    n_reduction_map = n_reduction_map.to(torch.float32)

                # Mapping functions
                mn_to_reduced = lambda v: torch.mm(m_reduction_map, torch.mm(v, torch.t(n_reduction_map))) # noqa
                mn_to_reduced_inv = lambda v: torch.mm(torch.t(m_reduction_map), torch.mm(v, n_reduction_map)) # noqa

                mm_to_reduced = lambda v: torch.mm(m_reduction_map, torch.mm(v, torch.t(m_reduction_map))) # noqa
                m1_to_reduced = lambda v: torch.mm(m_reduction_map, v)

                nn_to_reduced = lambda v: torch.mm(n_reduction_map, torch.mm(v, torch.t(n_reduction_map))) # noqa
                n1_to_reduced = lambda v: torch.mm(n_reduction_map, v)

                # Shrink
                # TODO: Im and In can instead be constants
                old_values = (F, Kx, Ky, Im, In, Lambda, Mu, S)
                F = mn_to_reduced(F)

                Kx = mm_to_reduced(Kx)
                Im = m1_to_reduced(Im)
                Mu = m1_to_reduced(Mu)

                Ky = nn_to_reduced(Ky)
                In = n1_to_reduced(In)
                Lambda = n1_to_reduced(Lambda)
                S = n1_to_reduced(S)

                timer.log('Shrink')

            if self.integration_type == 'MultiOmics':
                # Simplified grad with casting (atol=2e-6)
                # 100k iterations: 5.84e-5, 5.77e-5, 5.76e-5
                FKy = torch.mm(F, Ky)
                grad = (
                    4 * torch.mm(FKy, torch.mm(torch.t(F), FKy))
                    - 4 * a * torch.mm(Kx, FKy)
                    + torch.mm(Mu, torch.t(In))
                    + torch.mm(Im, torch.t(Lambda))
                    + self.rho * (
                        # Faster to multiply than to cast
                        torch.mm(F, torch.mm(In, torch.t(In)))
                        + torch.mm(
                            Im,
                            torch.mm(torch.t(Im), F)
                            + torch.t(S - 2 * In)
                        )
                    )
                )

                # # Simplified grad (atol=1e-6)
                # # 100k iterations: 6.16e-5, 6.08e-5, 6.08e-5
                # FKy = torch.mm(F, Ky)
                # grad = (
                #     4 * torch.mm(FKy, torch.mm(torch.t(F), FKy))
                #     - 4 * a * torch.mm(Kx, FKy)
                #     + torch.mm(Mu, torch.t(In))
                #     + torch.mm(Im, torch.t(Lambda))
                #     + self.rho * (
                #         # In * In.t is just nxn mat filled with 1
                #         torch.mm(F, torch.mm(In, torch.t(In)))
                #         # Factor Im
                #         - torch.mm(Im, torch.t(In))
                #         + torch.mm(Im, torch.mm(torch.t(Im), F))
                #         + torch.mm(Im, torch.t(S-In))
                #     )
                # )

                # # Original grad
                # # 100k iterations: 6.59e-5, 6.54e-5, 6.57e-5
                # original_grad = (
                #     4*torch.mm(F, torch.mm(Ky, torch.mm(torch.t(F), torch.mm(F, Ky))))
                #     - 4*a*torch.mm(Kx, torch.mm(F, Ky))
                #     + torch.mm(Mu, torch.t(In))
                #     + torch.mm(Im, torch.t(Lambda))
                #     + self.rho * (
                #         torch.mm(F, torch.mm(In, torch.t(In)))
                #         - torch.mm(Im, torch.t(In))
                #         + torch.mm(Im, torch.mm(torch.t(Im), F))
                #         + torch.mm(Im, torch.t(S-In))
                #     )
                # )
                #
                # if not torch.allclose(grad, original_grad, atol=2e-6):
                #     print(f'Max grad was  {torch.max(torch.abs(original_grad))}')
                #     print(f'Mean grad was {torch.mean(torch.abs(original_grad))}')
                #     assert False, f'Error was {torch.max(torch.abs(grad - original_grad))}'
            else:
                # TODO: Optimization for other project methods
                grad = (
                    dist + torch.mm(Im, torch.t(Lambda))
                    + self.rho * (
                        torch.mm(F, torch.mm(In, torch.t(In)))
                        - torch.mm(Im, torch.t(In))
                        + torch.mm(Im, torch.mm(torch.t(Im), F))
                        + torch.mm(Im, torch.t(S-In))
                    )
                )
            timer.log('Gradient Calculation')

            if self.gradient_reduction is not None:
                # Expand values
                F, Kx, Ky, Im, In, Lambda, Mu, S = old_values
                grad = mn_to_reduced_inv(grad)
                timer.log('Expand')

            i += 1
            Fst_moment = pho1 * Fst_moment + (1 - pho1) * grad
            Snd_moment = pho2 * Snd_moment + (1 - pho2) * grad * grad
            hat_Fst_moment = Fst_moment / (1 - np.power(pho1, i))
            hat_Snd_moment = Snd_moment / (1 - np.power(pho2, i))
            grad = hat_Fst_moment / (torch.sqrt(hat_Snd_moment) + delta)
            F_tmp = F - grad
            F_tmp[F_tmp < 0] = 0
            timer.log('Moment Calculation')

            # update
            F = (1 - self.epsilon) * F + self.epsilon * F_tmp
            timer.log('Update F')

            # update slack variable
            grad_s = Lambda + self.rho*(torch.mm(torch.t(F), Im) - In + S)
            s_tmp = S - grad_s
            s_tmp[s_tmp < 0] = 0
            S = (1-self.epsilon)*S + self.epsilon*s_tmp

            timer.log('Update Slack')

            # update dual variables
            Mu = Mu + self.epsilon*(torch.mm(F, In) - Im)
            Lambda = Lambda + self.epsilon*(torch.mm(torch.t(F), Im) - In + S)
            timer.log('Update Dual')

            # if scaling factor changes too fast, we can delay the update
            if self.integration_type == "MultiOmics":
                if i >= self.delay:
                    a = (
                        torch.trace(torch.mm(Kx, torch.mm(torch.mm(F, Ky), torch.t(F))))
                        / torch.trace(torch.mm(Kx, Kx))
                    )

            if (i+1) % self.log_pd == 0:
                if self.integration_type == "MultiOmics":
                    norm2 = torch.norm(a*Kx - torch.mm(torch.mm(F, Ky), torch.t(F)))
                    print("epoch:[{:d}/{:d}] err:{:.4f} alpha:{:.4f}"
                          .format(i+1, epochs, norm2.data.item(), a))
                else:
                    norm2 = torch.norm(dist*F)
                    print("epoch:[{:d}/{:d}] err:{:.4f}"
                          .format(i+1, epochs, norm2.data.item()))
            timer.log('Delay and CLI')

        if self.prime_dual_verbose:
            timer.aggregate()

        return F

    def project_nlma(self, dataset, F_list):
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

        # Set up manifold
        dim = len(F_list)
        W = F_list

        # TODO: Verify coef structure for >2 modalities
        print('Constructing W')
        coef = (1-mu) * torch.ones((dim, dim))
        coef = coef.fill_diagonal_(mu)
        for i, j in product(*(2 * [range(dim)])):
            W[i][j] *= coef[i, j]
        W = torch.from_numpy(np.bmat(W)).float().to(self.device)

        print('Computing Laplacian')
        L = laplacian(W)

        # Perform decomposition
        print('Converting to sparse matrix')
        L = sparse.csr_matrix(L)

        print('Calculating eigenvectors')
        # vals, vecs = np.linalg.eig(L)
        vals, vecs = sparse.linalg.eigs(L)

        print('Filtering eigenvectors')
        idx = np.argsort(vals)
        for i in range(len(idx)):
            if vals[idx[i]] >= eps:
                break
        vecs = vecs.real[:, idx[i:]]
        if vec_func:
            vecs = vec_func(vecs)

        for i in range(vecs.shape[1]):
            vecs[:, i] /= np.linalg.norm(vecs[:, i])

        print('Perfoming mapping')
        maps = []
        min_idx = 0
        for data in dataset:
            dx = data.shape[0]
            map = vecs[min_idx:min_idx + dx, :self.output_dim]
            maps.append(map)
            min_idx += dx

        return tuple(maps)
