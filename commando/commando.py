from itertools import product
import time
import warnings

import anndata as ad
import numpy as np
from scipy import linalg, sparse, stats
from scipy.sparse import linalg as sp_linalg
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
        two_step_aggregation_kwargs=None,
        two_step_log_pd=None,
        two_step_num_partitions=None,
        two_step_redundancy=1,
        two_step_pd_large=None,
        two_step_include_large=True,
        **kwargs,
    ):
        if 'project_mode' not in kwargs:
            kwargs['project_mode'] = 'nlma'

        self.gradient_reduction = gradient_reduction
        self.gradient_reduction_threshold = gradient_reduction_threshold

        self.two_step_pd_large = two_step_pd_large
        self.two_step_include_large = two_step_include_large
        self.two_step_aggregation = two_step_aggregation
        self.two_step_aggregation_kwargs = two_step_aggregation_kwargs
        self.two_step_num_partitions = two_step_num_partitions
        self.two_step_redundancy = two_step_redundancy

        self.prime_dual_verbose_timer = prime_dual_verbose_timer

        super().__init__(**kwargs)

        self.construct_sparse = (not self.two_step_include_large)

        self.two_step = (
            self.two_step_num_partitions is not None
            or two_step_aggregation in ['cell_cycle']
        )
        if self.two_step:
            self.dist_large = []
        if (
            two_step_log_pd is None
            and self.two_step
            and self.two_step_aggregation not in ['cell_cycle']
        ):
            self.two_step_log_pd = max(1, int(self.two_step_num_partitions / 10))
        else:
            self.two_step_log_pd = two_step_log_pd

    def fit_transform(self, dataset=None):
        """Fit function with ``nlma`` added"""
        distance_modes = [
            'euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis',
            'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming',
            'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean',
            'haversine',
        ]
        additional_distance_modes = [
            'geodesic', 'spearman', 'pearson',
        ]

        if self.integration_type not in ['MultiOmics']:
            raise Exception('integration_type error! Enter MultiOmics.')
        if self.distance_mode not in distance_modes + additional_distance_modes:
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

        # Test for dataset type (must all be the same)
        self.dataset = dataset
        self.dataset_annotation = None
        if isinstance(dataset[0], ad._core.anndata.AnnData):
            self.dataset = [d.X for d in self.dataset]
            self.dataset_annotation = dataset

        self.dataset_num = len(self.dataset)
        for i in range(self.dataset_num):
            self.row.append(np.shape(self.dataset[i])[0])
            self.col.append(np.shape(self.dataset[i])[1])

        if self.project_mode == 'nlma' and not (np.array(self.row) == self.row[0]).all():
            raise Exception("project_mode: 'nlma' requres aligned datasets.")

        if (not self.two_step) or self.two_step_redundancy == 1:
            # Create groupings (two-step)
            self.generate_group_idx()

            # Compute the distance matrix
            self.compute_distances()

            # Find correspondence between samples
            match_result = self.match()

            #  Project to common embedding
            integrated_data = self.project_nlma(
                match_result,
                compressed=(self.two_step)
            )

            # Unsort data
            if self.two_step:
                integrated_data = tuple(
                    integrated_data[i][self.idx_all_inv]
                    for i in range(self.dataset_num)
                )
        else:
            # ASDDF: Optimize this (distance calc, use compressed representation, etc.)
            match_result = None
            for i in range(self.two_step_redundancy):
                print(f'\nBeginning redundant step {i+1}')
                # Create groupings (two-step)
                self.generate_group_idx()

                # Compute the distance matrix
                self.compute_distances()

                # Find correspondence between samples
                if match_result is None:
                    match_result = self.compressed_to_dense(self.match())
                    for i, j in (
                        (i, j) for i in range(self.dataset_num) for j in range(self.dataset_num-i)
                    ):
                        match_result[i][j] = (
                            (1/self.two_step_redundancy)
                            * match_result[i][j][self.idx_all_inv][:, self.idx_all_inv]
                        )
                else:
                    # ASDDF: Revise aggregation method
                    match_output = self.compressed_to_dense(self.match())
                    for i, j in (
                        (i, j) for i in range(self.dataset_num) for j in range(self.dataset_num-i)
                    ):
                        match_result[i][j] += (
                            (1/self.two_step_redundancy)
                            * match_output[i][j][self.idx_all_inv][:, self.idx_all_inv]
                        )

                # Reset dataset order
                for i in range(self.dataset_num):
                    self.dataset[i] = self.dataset[i][self.idx_all_inv]

            #  Project to common embedding
            integrated_data = self.project_nlma(match_result)

        print('-' * 33)
        print('ComManDo Done!')
        time2 = time.time()
        print('Time:', time2-time1, 'seconds')

        return integrated_data

    def project_nlma(self, F_list, compressed=False):
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

        # ASDDF: Assess F value range for use as a simulated correlation matrix
        if not compressed:
            print('Constructing Dense W')
            W = [[None for j in range(self.dataset_num)] for i in range(self.dataset_num)]
            for i, j in (
                (i, j) for i in range(self.dataset_num) for j in range(self.dataset_num-i)
            ):
                if self.construct_sparse:
                    W[i][i+j] = F_list[i][j]
                    if i != j:
                        W[i+j][i] = W[i][i+j].transpose()
                else:
                    W[i][i+j] = F_list[i][j].cpu()
                    if i != j:
                        W[i+j][i] = torch.t(W[i][i+j])

        print('Applying Coefficients')
        # NOTE: Also includes final W assembly if not compressed
        coef_func = lambda a, b: mu if a == b else 1 - mu
        if compressed:
            for i, j in (
                (i, j) for i in range(self.dataset_num) for j in range(self.dataset_num-i)
            ):
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
            for i, j in product(*(2 * [range(self.dataset_num)])):
                coef = coef_func(i, j)
                W[i][j] *= coef
            if self.construct_sparse:
                W = sparse.bmat(W)
            else:
                W = torch.from_numpy(np.bmat(W)).float().cpu()

        print('Computing Laplacian')
        if compressed:
            # Calculate in compressed representation
            # Zero diagonal
            for i in range(self.dataset_num):
                for F_diag in F_list[i][0][0]:
                    dim = F_diag.shape[0]
                    F_diag.view(-1)[::dim + 1] = 0

            # Invert values
            for i, j in (
                (i, j) for i in range(self.dataset_num) for j in range(self.dataset_num-i)
            ):
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
            for col in range(self.dataset_num):
                running_col_sum = torch.zeros(self.row[0]).float().to(self.device)
                for row in range(self.dataset_num):
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
            for i in range(self.dataset_num):
                for F_diag in F_list[i][0][0]:
                    dim = F_diag.shape[0]
                    F_diag.view(-1)[::dim + 1] = new_diag[start_idx:start_idx+dim]
                    start_idx += dim

            print('Constructing L')
            L = self.assemble_matrix(F_list)
        elif self.construct_sparse:
            L = sparse.csgraph.laplacian(W)
        else:
            # Dense calculation of scipy.sparse.csgraph.laplacian from
            # ManiNetCluster
            n_nodes = self.row[0]
            L = -np.asarray(W)
            L.flat[::n_nodes + 1] = 0
            d = -L.sum(axis=0)
            L.flat[::n_nodes + 1] = d

        print('Calculating eigenvectors')
        if self.construct_sparse:
            vals, vecs = sp_linalg.eigsh(
                L,
                # ASDF: Replace this
                k=min((L.shape[0] - 1), 3*self.output_dim),
                which='SM',
                # ASDDDF: Make configurable
                tol=1e-4,
            )
        else:
            # ASDDF: Find way to calculate in compressed representation
            # to avoid L construction
            vals, vecs = linalg.eigh(
                L,
                lower=False,
                overwrite_a=True,
                # ASDDDF: Add options for user
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
                # Note: Default of 'evx' is only best when taking very few eigenvectors
                # from a large matrix
                driver='evx',
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
        for dx in self.row:
            map = vecs[min_idx:min_idx + dx, :self.output_dim]
            maps.append(map)
            min_idx += dx

        return tuple(maps)

    def match(self):
        """Find correspondence between multi-omics datasets"""
        print('Device:', self.device)
        cor_pairs = [[] for i in range(self.dataset_num)]
        for i in range(self.dataset_num):
            for j in range(i, self.dataset_num):
                print('-' * 33)
                print(f'Find correspondence between Dataset {i + 1} and Dataset {j + 1}')
                if self.two_step:
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
                    if self.two_step_include_large:
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

                        # ASDDF: How should this be handled?
                        use_diagonal = False
                        if use_diagonal:
                            for diag, rep_diag in zip(F_diag, F_rep.diag()):
                                diag += rep_diag
                        F_rep = F_rep.fill_diagonal_(0)

                    else:
                        F_rep = None

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

    def generate_group_idx(self):
        """Helper function to generate/refresh group idx"""
        if self.two_step:
            if self.two_step_num_partitions == 1:
                warnings.warn('``two_step_num_partitions`` = 1 can lead to unexpected behavior.')
            if self.two_step_redundancy < 1:
                raise Exception(
                    f'two_step_redundancy: {self.two_step_redundancy} is not supported'
                )

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
            ``idx_all_inv`` (automatic)
            [2,3,1,0]
            ``idx_groups``
            [[3,2],[0,1]]
            ``idx_sorted_groups``
            [[0,1],[2,3]]
            """
            # Assumes datasets are of the same size (NLMA)
            if self.two_step_aggregation == 'random':
                # Random sampling will generally lead to similar means,
                # potentially leading to large F collapse
                self.idx_all = np.arange(self.row[0])
                np.random.shuffle(self.idx_all)
                self.idx_groups = np.array_split(self.idx_all, self.two_step_num_partitions)
                self.idx_sorted_groups = np.array_split(
                    np.arange(self.row[0]),
                    self.two_step_num_partitions,
                )

            elif self.two_step_aggregation == 'cell_cycle':
                """
                Takes additional ``two_step_aggregation_kwargs``
                ['s_genes', 'g2m_genes']
                """
                assert self.dataset_annotation is not None, \
                    'Datasets must be given as type ``AnnData``.'
                assert all(
                    k in self.two_step_aggregation_kwargs
                    for k in ['s_genes', 'g2m_genes']
                ), (
                    "['s_genes', 'g2m_genes'] must be provided in "
                    "``two_step_aggregation_kwargs`` for use in "
                    "two_step_aggregation: 'cell_cycle'"
                )

                from scanpy.tl import score_genes_cell_cycle

                score_genes_cell_cycle(
                    self.dataset_annotation[0],
                    self.two_step_aggregation_kwargs['s_genes'],
                    self.two_step_aggregation_kwargs['g2m_genes'],
                    copy=False,
                )

                phase = np.array(self.dataset_annotation[0].obs['phase'])
                self.idx_all = np.argsort(phase)
                self.idx_groups = [
                    np.arange(self.row[1])[phase == p]
                    for p in np.unique(phase)
                ]
                self.idx_sorted_groups = [
                    np.arange(self.row[1])[np.sort(phase) == p]
                    for p in np.unique(phase)
                ]

                self.two_step_num_partitions = len(np.unique(phase))
                if self.two_step_log_pd is None:
                    self.two_step_log_pd = self.two_step_num_partitions

            elif self.two_step_aggregation == 'kmed':
                if self.two_step_redundancy < 1:
                    raise Exception(
                        f'two_step_redundancy: {self.two_step_redundancy} > 1 is not supported '
                        f'for two_step_aggregation: {self.two_step_aggregation}.'
                    )

                from sklearn_extra.cluster import KMedoids

                # ASDDF: Take both datasets into account
                # ASDDDF: Add distance metric changeability
                cluster = KMedoids(
                    n_clusters=self.two_step_num_partitions,
                    random_state=self.manual_seed,
                    method='alternate',
                    init='k-medoids++',
                ).fit(self.dataset[0])
                labels = cluster.labels_
                self.idx_all = np.argsort(labels)
                self.idx_groups = [
                    np.arange(self.row[0])[labels == i]
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

            # Auto-calculated
            self.idx_all_inv = np.argsort(self.idx_all)

            # Print group sizes (for skew debugging)
            print('Two-Step group sizes')
            group_len = [len(group) for group in self.idx_sorted_groups]
            print(f'Min: {min(group_len)}\nMax: {max(group_len)}')

            # Create dataset transforms
            self.rep_transform = (
                torch.zeros((self.row[0], self.two_step_num_partitions))
                .float().to(self.device)
            )
            for k, idx in enumerate(self.idx_groups):
                self.rep_transform[idx, k] = 1
            self.sparse_transform = sparse.csr_matrix(self.rep_transform.cpu())

            # Sort dataset
            for i in range(self.dataset_num):
                self.dataset[i] = self.dataset[i][self.idx_all]

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
                # ASDDDF: Compatibility with dense matrices
                def distance_function(df):
                    if df.shape[0] == 1:
                        return np.array([0])
                    distances, _ = stats.spearmanr(df.toarray(), axis=1)
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

            if self.two_step:
                # TODO: Add non-average aggregation methods
                # Intra-group distances
                self.dist.append([])
                for idx in self.idx_sorted_groups:
                    distances = distance_function(self.dataset[i][idx])
                    self.dist[-1].append(
                        torch.from_numpy(distances).float().to(self.device)
                    )

                # Inter-group distances
                if self.two_step_include_large:
                    agg_intra = normalize(self.sparse_transform).transpose() * self.dataset[i]
                    agg_intra = distance_function(agg_intra)
                    self.dist_large.append(
                        torch.from_numpy(agg_intra).float().to(self.device)
                    )
            else:
                distances = distance_function(self.dataset[i])
                self.dist.append(torch.from_numpy(distances).float().to(self.device))

    def compressed_to_dense(self, F_list):
        """
        Convert compressed F_list to single partition representation
        (equivalent to ``self.two_step_num_partitions`` None)

        Output
        ------
        F_list containing all F matrices
        """
        for i, j in ((i, j) for i in range(self.dataset_num) for j in range(self.dataset_num-i)):
            F_diag, F_rep = F_list[i][j]
            if self.construct_sparse:
                # ASDDF: GPU fixes
                F_list[i][j] = sparse.block_diag(F_diag, format='csr')
            else:
                dense = torch.block_diag(*F_diag) + self.expand_matrix(F_rep)
                F_list[i][j] = dense.cpu()
        return F_list

    def assemble_matrix(self, F_list):
        """
        Assemble full matrix

        Output
        ------
        Single matrix combining all F matrices
        """
        W = [[None for j in range(self.dataset_num)] for i in range(self.dataset_num)]
        for i, j in ((i, j) for i in range(self.dataset_num) for j in range(self.dataset_num-i)):
            F_diag, F_rep = F_list[i][j]
            if self.construct_sparse:
                # ASDDF: GPU fixes
                W[i][i+j] = sparse.block_diag(F_diag)
                if i != j:
                    W[i+j][i] = W[i][i+j].transpose()
            else:
                W[i][i+j] = torch.block_diag(*F_diag) + self.expand_matrix(F_rep)
                W[i][i+j] = W[i][i+j].cpu()
                if i != j:
                    W[i+j][i] = torch.t(W[i][i+j])

        if self.construct_sparse:
            return sparse.bmat(W)
        else:
            return torch.from_numpy(np.bmat(W)).float().cpu()

    def expand_matrix(self, m):
        """Helper function to cast two_step large matrix to its final size"""
        return torch.mm(
            self.rep_transform,
            torch.mm(
                m,
                torch.t(self.rep_transform),
            ),
        )

    def expand_matrix_sparse(self, m):
        """Helper function to cast sparse two_step large matrix to its final size"""
        return (
            self.sparse_transform
            * m
            * self.sparse_transform.transpose()
        )
