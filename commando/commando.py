from itertools import product
import time
import warnings

import numpy as np
from scipy import linalg, sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
import torch
import unioncom.UnionCom as uc
from unioncom.utils import geodesic_distances, init_random_seed

from .utilities import time_logger


class ComManDo(uc.UnionCom):
    """Adaptation of https://github.com/caokai1073/UnionCom by caokai1073"""
    def __init__(
        self,
        gradient_reduction=None,
        gradient_reduction_threshold=.99,
        prime_dual_verbose_timer=False,
        two_step_aggregation='random',
        two_step_log_pd=None,
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

        self.prime_dual_verbose_timer = prime_dual_verbose_timer

        super().__init__(**kwargs)

        if self.two_step_num_partitions is not None:
            self.dist_large = []
        if two_step_log_pd is None and self.two_step_num_partitions is not None:
            self.two_step_log_pd = max(1, int(self.two_step_num_partitions / 10))
        else:
            self.two_step_log_pd = two_step_log_pd

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

        if self.integration_type not in ['MultiOmics']:
            raise Exception('integration_type error! Enter MultiOmics.')
        if self.distance_mode != 'geodesic' and self.distance_mode not in distance_modes:
            raise Exception('distance_mode error! Enter a correct distance_mode.')
        # ASDDDF: Readd ('tsne', 'barycentric')
        if self.project_mode not in ('nlma'):
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

        # Create groupings (two-step)
        # ASDF: Non-random, potentially OrthoClust
        if self.two_step_num_partitions is not None:
            if self.two_step_num_partitions == 1:
                warnings.warn('``two_step_num_partitions`` = 1 can lead to unexpected behavior.')

            self.idx_all = np.arange(self.row[i])

            """
            Each method modifies a few vars

            ``idx_all`` is the idx to sort the data (into group order)
            ``idx_all_inv`` is the idx to unsort the data
            ``idx_groups`` is the idx to access each group in the unsorted array
            ``idx_sorted_groups`` is the idx to access each group in the sorted array

            Ex.
            We have group indexes [[3,2],[0,1]]
            ``idx_all``
            [3,2,0,1]
            ``idx_all_inv``
            [2,3,1,0]
            ``idx_groups``
            [[3,2],[0,1]]
            ``idx_sorted_groups``
            [[0,1],[2,3]]
            """
            if self.two_step_aggregation == 'random':
                # Random sampling will generally lead to similar means,
                # leading to large F collapse
                np.random.shuffle(self.idx_all)
                self.idx_groups = np.array_split(self.idx_all, self.two_step_num_partitions)
                self.idx_sorted_groups = np.array_split(
                    np.arange(self.row[0]),
                    self.two_step_num_partitions,
                )
                self.idx_all_inv = np.argsort(self.idx_all)

            elif self.two_step_aggregation == 'kmed':
                from sklearn_extra.cluster import KMedoids

                # ASDDF: Take both datasets into account
                # ASDDDF: Add distance metric changeability
                cluster = KMedoids(
                    n_clusters=self.two_step_num_partitions,
                    random_state=self.manual_seed,
                    method='alternate',
                    init='k-medoids++',
                ).fit(dataset[0])
                labels = cluster.labels_
                self.idx_all = np.argsort(labels)
                self.idx_all_inv = np.argsort(self.idx_all)
                self.idx_groups = [
                    np.arange(self.row[1])[labels == i]
                    for i in range(self.two_step_num_partitions)
                ]
                running_idx = []
                for group in self.idx_groups[:-1]:
                    if len(running_idx) == 0:
                        running_idx.append(len(group))
                    else:
                        running_idx.append(len(group) + running_idx[-1])
                self.idx_sorted_groups = np.split(np.arange(self.row[0]), running_idx)
            else:
                raise Exception(
                    '``two_step_aggregation`` type '
                    f"'{self.two_step_aggregation}' not found"
                )

            # Print group sizes (for skew debugging)
            print('Two-Step group sizes:')
            for group in self.idx_sorted_groups[:-1]:
                print(len(group), end=', ')
            print(len(self.idx_sorted_groups[-1]))

            # Create dataset transforms
            self.rep_transform = (
                torch.zeros((self.row[i], self.two_step_num_partitions))
                .float().to(self.device)
            )
            for k, idx in enumerate(self.idx_groups):
                self.rep_transform[idx, k] = 1
            self.sparse_transform = sparse.csr_matrix(self.rep_transform.cpu())

            # Sort dataset
            for i in range(dataset_num):
                dataset[i] = dataset[i][self.idx_all]

        # Compute the distance matrix
        print('Shape of Raw data')
        for i in range(dataset_num):
            print('Dataset {}:'.format(i), np.shape(dataset[i]))
            dataset[i] = (
                (dataset[i] - np.min(dataset[i]))
                / (np.max(dataset[i]) - np.min(dataset[i]))
            )

            if self.distance_mode == 'geodesic':
                def distance_function(df):
                    distances = geodesic_distances(df, self.kmax)
                    return np.array(distances)
            else:
                def distance_function(df):
                    return pairwise_distances(df, metric=self.distance_mode)

            if (self.two_step_num_partitions is not None):
                # Intra-group distances
                self.dist.append([])
                for idx in self.idx_sorted_groups:
                    distances = distance_function(dataset[i][idx])
                    self.dist[-1].append(
                        torch.from_numpy(distances).float().to(self.device)
                    )

                # Inter-group distances
                agg_intra = normalize(self.sparse_transform).transpose() * dataset[i]
                agg_intra = distance_function(agg_intra)
                self.dist_large.append(
                    torch.from_numpy(agg_intra).float().to(self.device)
                )
            else:
                distances = distance_function(dataset[i])
                self.dist.append(torch.from_numpy(distances).float().to(self.device))

        # Find correspondence between samples
        match_result = self.match(dataset=dataset)

        #  Project to common embedding
        integrated_data = self.project_nlma(dataset, match_result)

        # Unsort data
        if self.two_step_num_partitions is not None:
            integrated_data = tuple(
                integrated_data[i][self.idx_all_inv]
                for i in range(dataset_num)
            )

        print('-' * 33)
        print('ComManDo Done!')
        time2 = time.time()
        print('Time:', time2-time1, 'seconds')

        return integrated_data

    def match(self, dataset):
        """Find correspondence between multi-omics datasets"""
        print('Device:', self.device)
        dataset_num = len(dataset)

        cor_pairs = [[] for i in range(dataset_num)]
        for i in range(dataset_num):
            for j in range(i, dataset_num):
                print('-' * 33)
                print(f'Find correspondence between Dataset {i + 1} and Dataset {j + 1}')
                if self.two_step_num_partitions is not None:
                    # First step (small F)
                    F_diag = []
                    for k, (i_dist, j_dist) in enumerate(zip(self.dist[i], self.dist[j])):
                        small_f_verbose = (k+1) % self.two_step_log_pd == 0
                        if small_f_verbose:
                            print(f'Calculating intra-group F #{k + 1}')
                        F_diag.append(self.Prime_Dual(
                            [i_dist, j_dist],
                            dx=self.col[i],
                            dy=self.col[j],
                            verbose=small_f_verbose,
                        ))

                    # # Reconstruct and unsort large F
                    # print('Constructing large F')
                    # F = torch.block_diag(*F_diag)

                    # Second step (Large F)
                    print('Calculating inter-group F')
                    i_dist = self.dist_large[i]
                    j_dist = self.dist_large[j]

                    # Calculate inter-pseudo F
                    dist = [i_dist, j_dist]
                    F_rep = self.Prime_Dual(
                        dist,
                        dx=self.col[i],
                        dy=self.col[j],
                        epoch_override=self.two_step_pd_large,
                    )

                    # # ASDDF: Is this line justified?
                    F_rep = F_rep.fill_diagonal_(0)

                    # ASDDF: Store only upper triangular (?)
                    # ASDF: Apply scaling factors for K_x ~= FK_yF^t compatibility
                    F = (F_diag, F_rep)
                else:
                    F = self.Prime_Dual(
                        [self.dist[i], self.dist[j]],
                        dx=self.col[i],
                        dy=self.col[j],
                    )
                cor_pairs[i].append(F)

        print("Finished Matching!")
        return cor_pairs

    def Prime_Dual(
        self,
        dist,
        dx=None,
        dy=None,
        epoch_override=None,
        verbose=True,
    ):
        """Prime dual combined with Adam algorithm to find the local optimal soluation"""
        Kx = dist[0]
        Ky = dist[1]

        # Escape for 1x1 data
        if Kx.shape == (1, 1) and Ky.shape == (1, 1):
            if verbose:
                print('1x1 distance matrix, escaping...')
            return torch.ones((1, 1)).float().to(self.device)

        N = np.int(np.maximum(Kx.shape[0], Ky.shape[0]))
        Kx = Kx / N
        Ky = Ky / N
        Kx = Kx.float().to(self.device)
        Ky = Ky.float().to(self.device)
        a = np.sqrt(dy/dx)
        m = np.shape(Kx)[0]
        n = np.shape(Ky)[0]

        # F = np.zeros((m, n))
        F = torch.zeros((m, n)).float().to(self.device)
        Im = torch.ones((m, 1)).float().to(self.device)
        # Imm = torch.ones((m, m)).float().to(self.device)
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
        timer = time_logger(record=self.prime_dual_verbose_timer)

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
                # ASDDDF: Im and In can instead be constants
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

            # # Simplified grad with casting (atol=2e-6)
            # # 100k iterations: 5.84e-5, 5.77e-5, 5.76e-5
            # FKy = torch.mm(F, Ky)
            # grad = (
            #     4 * torch.mm(FKy, torch.mm(torch.t(F), FKy))
            #     - 4 * a * torch.mm(Kx, FKy)
            #     + torch.mm(Mu, torch.t(In))
            #     + torch.mm(Im, torch.t(Lambda))
            #     + self.rho * (
            #         # Faster to multiply than to cast/repeat
            #         torch.mm(F, torch.mm(In, torch.t(In)))
            #         + torch.mm(
            #             Im,
            #             torch.t(S - 2 * In)
            #         )
            #         + torch.mm(Imm, F)
            #     )
            # )

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
            # if not torch.allclose(grad, original_grad, atol=1e-5):
            #     print(f'Max grad was  {torch.max(torch.abs(original_grad))}')
            #     print(f'Mean grad was {torch.mean(torch.abs(original_grad))}')
            #     assert False, f'Error was {torch.max(torch.abs(grad - original_grad))}'
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

            if verbose and (i+1) % self.log_pd == 0:
                if self.integration_type == "MultiOmics":
                    norm2 = torch.norm(a*Kx - torch.mm(torch.mm(F, Ky), torch.t(F)))
                    print("epoch:[{:d}/{:d}] err:{:.4f} alpha:{:.4f}"
                          .format(i+1, epochs, norm2.data.item(), a))
                else:
                    norm2 = torch.norm(dist*F)
                    print("epoch:[{:d}/{:d}] err:{:.4f}"
                          .format(i+1, epochs, norm2.data.item()))
            timer.log('Delay and CLI')

        if self.prime_dual_verbose_timer:
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
        dataset_num = len(dataset)
        mu = .9  # ASDF: Make this lower
        eps = 1e-8
        vec_func = None

        # Set up manifold
        W = [[None for j in range(dataset_num)] for i in range(dataset_num)]

        # Dense F
        def expand_matrix(m):
            return torch.mm(
                self.rep_transform,
                torch.mm(
                    m,
                    torch.t(self.rep_transform),
                ),
            )

        if self.two_step_num_partitions is None:
            print('Constructing Dense W')
            for i, j in ((i, j) for i in range(dataset_num) for j in range(dataset_num-i)):
                W[i][i+j] = F_list[i][j].cpu()
                if i != j:
                    W[i+j][i] = torch.t(W[i][i+j])

        print('Applying Coefficients')
        coef_func = lambda a, b: mu if a == b else 1 - mu
        if self.two_step_num_partitions is not None:
            for i, j in ((i, j) for i in range(dataset_num) for j in range(dataset_num-i)):
                coef = coef_func(i, i+j)
                # Apply in compressed representation
                F_list[i][j] = (
                    [
                        coef * F_diag_component
                        for F_diag_component in F_list[i][j][0]
                    ],
                    coef * F_list[i][j][1]
                )
        else:
            for i, j in product(*(2 * [range(dataset_num)])):
                coef = coef_func(i, j)
                W[i][j] *= coef
            W = torch.from_numpy(np.bmat(W)).float().cpu()

        print('Computing Laplacian')
        if self.two_step_num_partitions is not None:
            # Calculate in compressed representation
            # Zero diagonal
            for i in range(dataset_num):
                for F_diag in F_list[i][0][0]:
                    dim = F_diag.shape[0]
                    F_diag.view(-1)[::dim + 1] = 0

            # Invert values
            for i, j in ((i, j) for i in range(dataset_num) for j in range(dataset_num-i)):
                F_diag, F_rep = F_list[i][j]
                for mat in (*F_diag, F_rep):
                    mat *= -1

            # Get degrees
            def get(upper_triangle_block, i, j):
                if j >= i:
                    return(upper_triangle_block[i][j-i])
                output = upper_triangle_block[j][i-j]
                return (
                    [torch.t(mat) for mat in output[0]],
                    torch.t(output[1])
                )

            new_diag = []
            for col in range(dataset_num):
                running_col_sum = torch.zeros(self.row[0]).float().to(self.device)
                for row in range(dataset_num):
                    F_diag, F_rep = get(F_list, row, col)
                    rep_col_sum = torch.sum(F_rep, 0)
                    total_col_sum = [
                        rep + torch.sum(mat, 0)
                        for mat, rep in zip(F_diag, rep_col_sum)
                    ]
                    running_col_sum += torch.cat(total_col_sum)
                running_col_sum *= -1
                new_diag.append(running_col_sum)
            new_diag = torch.cat(new_diag)

            # Reassign to diagonal
            start_idx = 0
            for i in range(dataset_num):
                for F_diag in F_list[i][0][0]:
                    dim = F_diag.shape[0]
                    F_diag.view(-1)[::dim + 1] = new_diag[start_idx:start_idx+dim]
                    start_idx += dim
        else:
            # Dense calculation of scipy.sparse.csgraph from
            # ManiNetCluster
            n_nodes = self.row[0]
            L = -np.asarray(W)
            L.flat[::n_nodes + 1] = 0
            d = -L.sum(axis=0)
            L.flat[::n_nodes + 1] = d

        if self.two_step_num_partitions is not None:
            print('Constructing L')
            for i, j in ((i, j) for i in range(dataset_num) for j in range(dataset_num-i)):
                # ASDF: Avoid needing to do this
                F_diag, F_rep = F_list[i][j]
                W[i][i+j] = torch.block_diag(*F_diag) + expand_matrix(F_rep)
                W[i][i+j] = W[i][i+j].cpu()
                if i != j:
                    W[i+j][i] = torch.t(W[i][i+j])
            L = torch.from_numpy(np.bmat(W)).float().cpu()

        print('Calculating eigenvectors')
        # ASDDF: Find way to calculate in compressed representation
        # vals, vecs = sp_linalg.eigsh(L, )
        vals, vecs = linalg.eigh(
            L,
            overwrite_a=True,
            # ASDDF: Find better way to subset by index
            # subset_by_index=(0, self.output_dim + 10),
            subset_by_value=(eps, np.inf),
        )

        # ASDDF: Shorten if eig solution supports filtering
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
