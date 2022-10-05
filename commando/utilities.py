import contextlib
import math
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
import torch
import torch.nn as nn
import umap


def outliers(x, leniency=1.5, aggregate=False, return_limits=False, verbose=False):
    """Detect outliers"""
    # https://stackoverflow.com/a/11886564
    # , threshold=3.5
    # med = np.median(x, axis=0)
    # d = np.linalg.norm(x - med, ord=2, axis=1)
    # med_d = np.median(d)
    # mod_z = .6745 * d / med_d
    # return mod_z > threshold

    # Box and whisker type
    Q1 = np.percentile(x, 25, axis=0, keepdims=True)
    Q2 = np.percentile(x, 50, axis=0, keepdims=True)
    Q3 = np.percentile(x, 75, axis=0, keepdims=True)
    span = Q3 - Q1
    lower_bound = Q1 - leniency * span
    upper_bound = Q3 + leniency * span
    if verbose:
        print(f'Lower: {lower_bound}')
        print(f'Upper: {upper_bound}')
    result = (x < lower_bound) + (x > upper_bound)
    if aggregate:
        result = np.prod(result, axis=1)
    if return_limits:
        return result, (lower_bound, upper_bound, span)
    return result


def identity(x):
    """Identity function, lambda cannot be used for pickle"""
    return x


def reduce_sample_data(df, num_samples=1000, num_features=1000):
    """Takes ``df``, a sparse matrix, and reduces features by std"""
    sample = df[:num_samples]
    std = sample.power(2).mean(axis=0) - np.power(sample.mean(axis=0), 2)
    ret_idx = np.squeeze(np.array(np.argsort(-std)))[:num_features]
    return df[:, ret_idx]


class time_logger():
    """Class made for easy logging with toggleable verbosity"""
    def __init__(
        self,
        discard_first_sample=False,
        record=True,
        verbose=False,
    ):
        self.discard_first_sample = discard_first_sample
        self.record = record
        self.verbose = verbose

        self.history = {}
        self.start_time = perf_counter()

    def log(self, str=''):
        """Print with message ``str`` if verbose.  Otherwise, skip"""
        if not (self.verbose or self.record):
            return  # Cut timing for optimization
        self.end_time = perf_counter()

        # Perform any auxiliary operations here
        time_elapsed = self.end_time - self.start_time
        if self.record:
            if str not in self.history:
                self.history[str] = []
            self.history[str].append(time_elapsed)
        if self.verbose:
            print(f'{str}: {time_elapsed}')

        # Re-time to avoid extra runtime cost
        self.start_time = perf_counter()

    def aggregate(self):
        """Print mean times for all keys in ``self.history``"""
        running_total = 0
        for k, v in self.history.items():
            avg_time_elapsed = np.array(v)
            if self.discard_first_sample:
                avg_time_elapsed = avg_time_elapsed[1:]
            avg_time_elapsed = np.mean(np.array(v))

            running_total += avg_time_elapsed
            print(f'{k}: {avg_time_elapsed}')
        print(f'Total: {running_total}')


def visualize_mapping(mapping, primary=0):
    """Visualize a mapping given as (*mappings) using PCA, first mapping is the primary one"""
    assert len(mapping) == 2, 'Currently, ``visualize_mapping`` only supports 2 mappings'

    pca = PCA(n_components=2)
    pca.fit(mapping[primary])
    for i, m in enumerate(mapping):
        m_pca = pca.transform(m)
        label = f'Mapping {i+1}'
        # ASDF: Make primary show first
        if i == primary:
            s, c = 20, 'orange'
        else:
            s, c = 2, 'blue'
        plt.scatter(m_pca[:, 0], m_pca[:, 1], label=label, s=s, c=c)
    plt.title('ComManDo PCA Plot')
    plt.legend(loc='best')


def uc_visualize(data, data_integrated, datatype=None, mode=None):
    """
    Visualize function with 'None' mapping mode added, modified from
    https://github.com/caokai1073/UnionCom
    """
    assert (mode in ['PCA', 'UMAP', 'TSNE'] or mode is None), (
        "Mode has to be one of 'PCA', 'UMAP', 'TSNE', or None."
    )

    # styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c', 'greenyellow', 'lightcoral', 'teal']
    # data_map = ['Chromatin accessibility', 'DNA methylation', 'Gene expression']
    # color_map = ['E5.5','E6.5','E7.5']
    dataset_num = len(data)

    def embed_data(data):
        if mode == 'PCA':
            return PCA(n_components=2).fit_transform(data)
        elif mode == 'TSNE':
            return TSNE(n_components=2).fit_transform(data)
        elif mode == 'UMAP':
            return umap.UMAP(n_components=2).fit_transform(data)
        else:
            return data[:, :2]

    embedding = []
    dataset_xyz = []
    for i in range(dataset_num):
        dataset_xyz.append("data{:d}".format(i+1))
        embedding.append(embed_data(data[i]))

    if mode == 'PCA':
        label_x, label_y = 'PCA-1', 'PCA-2'
    elif mode == 'TSNE':
        label_x, label_y = 'TSNE-1', 'TSNE-2'
    elif mode == 'UMAP':
        label_x, label_y = 'UMAP-1', 'UMAP-2'
    else:
        label_x, label_y = 'NONE-1', 'NONE-2'

    plt.figure()
    if datatype is not None:
        for i in range(dataset_num):
            plt.subplot(1, dataset_num, i+1)
            for j in set(datatype[i]):
                index = np.where(datatype[i] == j)
                plt.scatter(embedding[i][index, 0], embedding[i][index, 1], s=5.)
            plt.title(dataset_xyz[i])
            plt.xlabel(label_x)
            plt.ylabel(label_y)
    else:
        for i in range(dataset_num):
            plt.subplot(1, dataset_num, i+1)
            plt.scatter(embedding[i][:, 0], embedding[i][:, 1], s=5.)
            plt.title(dataset_xyz[i])
            plt.xlabel(label_x)
            plt.ylabel(label_y)
            plt.title(dataset_xyz[i])

    plt.tight_layout()

    data_all = np.vstack((data_integrated[0], data_integrated[1]))
    for i in range(2, dataset_num):
        data_all = np.vstack((data_all, data_integrated[i]))
    embedding_all = embed_data(data_all)

    tmp = 0
    num = [0]
    for i in range(dataset_num):
        num.append(tmp+np.shape(data_integrated[i])[0])
        tmp += np.shape(data_integrated[i])[0]

    embedding = []
    for i in range(dataset_num):
        embedding.append(embedding_all[num[i]:num[i+1]])

    color = [
        [1, 0.5, 0],
        [0.2, 0.4, 0.1],
        [0.1, 0.2, 0.8],
        [0.5, 1, 0.5],
        [0.1, 0.8, 0.2],
    ]
    # marker=['x','^','o','*','v']

    plt.figure()
    if datatype is not None:

        datatype_all = np.hstack((datatype[0], datatype[1]))
        for i in range(2, dataset_num):
            datatype_all = np.hstack((datatype_all, datatype[i]))

        plt.subplot(1, 2, 1)
        for i in range(dataset_num):
            plt.scatter(embedding[i][:, 0], embedding[i][:, 1], c=color[i], s=5., alpha=0.8)
        plt.title('Integrated Embeddings')
        plt.xlabel(label_x)
        plt.ylabel(label_y)

        plt.subplot(1, 2, 2)
        for j in set(datatype_all):
            index = np.where(datatype_all == j)
            plt.scatter(embedding_all[index, 0], embedding_all[index, 1], s=5., alpha=0.8)

        plt.title('Integrated Cell Types')
        plt.xlabel(label_x)
        plt.ylabel(label_y)

    else:

        for i in range(dataset_num):
            plt.scatter(embedding[i][:, 0], embedding[i][:, 1], c=color[i], s=5., alpha=0.8)
        plt.title('Integrated Embeddings')
        plt.xlabel(label_x)
        plt.ylabel(label_y)

    plt.tight_layout()
    plt.show()


def ensure_list(x):
    if not (isinstance(x, np.ndarray) or isinstance(x, type([]))):
        return np.array([x])
    return np.array(x)


class SimpleModel(nn.Module):
    """Thin, simple NN model"""
    def __init__(self, input_dim, output_dim, hidden_dim=16, p=0.6):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, *X):
        """Forward pass for the model"""
        return self.fc2(self.dropout(self.fc1(X[0])))

    def lastForward(self, *X):
        """Output forward pass for the model"""
        return self.fc2(self.fc1(X[0]))

    def loss(self, logits, *Y, criterion=None):
        """Loss for the model"""
        return criterion(logits, Y[1])


class SimpleDualModel(nn.Module):
    """Thin, simple NN model"""
    def __init__(self, input_dim, output_dim, hidden_dim=10, p=0.6):
        super().__init__()

        self.fc1_1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc1_2 = nn.Linear(hidden_dim, input_dim)

        self.fc2_1 = nn.Linear(output_dim, hidden_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc2_2 = nn.Linear(hidden_dim, output_dim)

        self.conv = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, *X):
        """Forward pass for the model"""
        e1, e2 = self.fc1_1(X[0]), self.fc2_1(X[1])
        return (
            self.fc1_2(self.dropout(e1)),
            self.fc2_2(self.dropout(e2)),
            self.conv(e1), e2)

    def lastForward(self, *X):
        """Output forward pass for the model"""
        return self.fc2_2(self.conv(self.fc1_1(X[0])))

    def loss(self, logits, *Y, criterion=None):
        """Loss for the model"""
        return (
            criterion(logits[0], Y[0])
            + criterion(logits[1], Y[1])
            + criterion(logits[2], logits[3].detach()))


class SimpleCommonDualModel(nn.Module):
    """Thin, simple NN model"""
    def __init__(self, input_dim, output_dim, hidden_dim=10, p=0.6):
        super().__init__()

        self.fc1_1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc1_2 = nn.Linear(hidden_dim, input_dim)

        self.fc2_1 = nn.Linear(output_dim, hidden_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc2_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, *X):
        """Forward pass for the model"""
        e1, e2 = self.fc1_1(X[0]), self.fc2_1(X[1])
        return (
            self.fc1_2(self.dropout(e1)),
            self.fc2_2(self.dropout(e2)),
            e1, e2)

    def lastForward(self, *X):
        """Output forward pass for the model"""
        return self.fc2_2(self.fc1_1(X[0]))

    def loss(self, logits, *Y, criterion=None):
        """Loss for the model"""
        return (
            criterion(logits[0], Y[0])
            + criterion(logits[1], Y[1])
            + criterion(logits[2], logits[3]))


class BABELMini(nn.Module):
    """Dual autoencoder based on BABEL"""
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super().__init__()

        self.fc1_1 = nn.Linear(input_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, input_dim)

        self.fc2_1 = nn.Linear(output_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, *X):
        """Forward pass for the model"""
        e1, e2 = self.fc1_1(X[0]), self.fc2_1(X[1])
        return (
            self.fc1_2(e1),
            self.fc2_2(e2),
            self.fc2_2(e1),
            self.fc1_2(e2))

    def lastForward(self, *X):
        """Output forward pass for the model"""
        return self.fc2_2(self.fc1_1(X[0]))

    def loss(self, logits, *Y, criterion=None):
        """Loss for the model"""
        return (
            criterion(logits[0], Y[0])
            + criterion(logits[1], Y[1])
            + criterion(logits[2], Y[1])
            + criterion(logits[3], Y[0]))


class SingleModel(nn.Module):
    """Thin, single-layer NN model"""
    def __init__(self, input_dim, output_dim, p=0.6):
        super().__init__()

        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, *X):
        """Forward pass for the model"""
        return self.fc1(self.dropout(X[0]))

    def lastForward(self, *X):
        """Output forward pass for the model"""
        return self.fc1(self.dropout(X[0]))

    def loss(self, logits, *Y, criterion=None):
        """Loss for the model"""
        return criterion(logits, Y[1])


def predict_knn(input, output, val=None, k=5):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(input, output)
    if val is not None:
        return knn.predict(val)
    return knn.predict(input)


def predict_nn(source, target, val=None, epochs=200, batch_size=32):
    """Predict modality using a simple NN"""
    model = SimpleCommonDualModel(source.shape[1], target.shape[1])
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.MSELoss()
    batches = int(len(source)/batch_size)

    epoch_str_len = len(str(epochs))
    batch_str_len = len(str(batches))
    loss_detached = 0.0
    for epoch in range(epochs):
        prog_str = math.floor(25*(epoch+1)/epochs) * '|'
        for batch in range(batches):
            batch_str = math.floor(10*(batch+1)/batches) * '+'
            print(
                f'{epoch+1:>{epoch_str_len}}/{epochs} [{prog_str:<25}]: - '
                f'{batch+1:>{batch_str_len}}/{batches} ({batch_str:<10}) - '
                f'Loss: {loss_detached:.4f}',
                end='\r')
            idx = np.random.choice(range(len(source)), batch_size, replace=False)
            optimizer.zero_grad()
            logits = model(source[idx], target[idx])
            loss = model.loss(logits, source[idx], target[idx], criterion=criterion)
            loss_detached = loss.detach()
            loss.backward()
            optimizer.step()
    print('\nDone!')
    if val is not None:
        return model.lastForward(val).detach().cpu().numpy()
    return model.lastForward(source).detach().cpu().numpy()


def set_yticks(ax, num_ticks):
    """Set a specified number of yticks in axes ax"""
    yrange = (ax.get_ylim()[1] - ax.get_ylim()[0])
    bottom = ax.get_ylim()[0] + .1 * yrange
    top = ax.get_ylim()[1] - .1 * yrange
    ax.set_yticks(np.round(np.linspace(bottom, top, num_ticks), 1))


def tune_cm(cm, dataset, types, wt_size, num_search=20):
    best_acc = 0
    wt_str = np.random.rand(wt_size * num_search)
    for i in range(num_search):
        wt = wt_str[wt_size*i:wt_size*(i+1)]

        with contextlib.redirect_stdout(None):
            cm.loss_weights = wt
            cm_data = cm.fit_transform(dataset=dataset)
            acc = cm.test_LabelTA(cm_data, types)

        if acc > best_acc:
            best_cm_data = cm_data
            best_acc = acc
            best_wt = wt
        print(f'Done:{100 * (i+1) / num_search:.1f}%; Max:{best_acc:.3f}; Curr:{acc:.3f}', end='\r')
    print()
    print(f'Best Weights: {best_wt}')
    return best_wt, best_cm_data


def sort_by_interest(datasets, int_thresh=.8, limit=20, remove_outliers=True):
    """Assesses datasets (real, imputed) and returns interesting indexes"""
    if limit is None:
        limit = datasets[0].shape[1]

    # Score by entropy and imputation performance
    # Distribution
    # X = np.linspace(0, 1, 1000)
    if remove_outliers:
        dataset0_features = [
            datasets[0][~outliers(datasets[0][:, i]), i]
            for i in range(datasets[0].shape[1])]
    else:
        dataset0_features = [datasets[0][:, i] for i in range(datasets[0].shape[1])]
    distribution_true = [
        np.histogram(d, bins=np.linspace(np.min(d), np.max(d), 100))[0] for d in dataset0_features
    ]
    # distribution_true = [stats.rv_histogram(d) for d in distribution_true]
    # distribution_true = [[d.pdf(x) for x in X] for d in distribution_true]
    # distribution_true = [d / sum(d) for d in distribution_true]
    distribution_pred = [
        np.histogram(
            datasets[1][:, i],
            bins=np.linspace(np.min(datasets[1][:, i]),
            np.max(datasets[1][:, i]), 100))[0]
            for i in range(datasets[0].shape[1])
    ]
    # distribution_pred = [stats.rv_histogram(d) for d in distribution_pred]
    # distribution_pred = [[d.pdf(x) for x in X] for d in distribution_pred]
    # distribution_pred = [d / sum(d) for d in distribution_pred]

    # Entropy
    entropy_arr = np.array([stats.entropy(t) for t in distribution_true])
    entropy_arr[np.isnan(entropy_arr)] = 0
    entropy_arr[np.isinf(entropy_arr)] = 0

    # Corr
    corr_arr = np.array([
        stats.pearsonr(datasets[0][:, i], datasets[1][:, i])[0]
        for i in range(datasets[0].shape[1])])
    corr_arr[np.isnan(corr_arr)] = -1

    # # MSE
    # dist_arr = np.array([
    #     np.mean(np.sum((datasets[0][:, i] - datasets[1][:, i])**2))
    #     for i in range(datasets[0].shape[1])])
    # dist_arr[np.isnan(dist_arr)] = np.inf

    # # KL
    # kl_arr = np.array([stats.entropy(t, p)
    #     for t, p in zip(distribution_true, distribution_pred)])
    # kl_arr[np.isnan(kl_arr)] = np.inf

    # # KS
    # ks_arr = np.array([stats.kstest(p, t)[0]
    #     for t, p in zip(distribution_true, distribution_pred)])
    # ks_arr[np.isnan(ks_arr)] = 1

    # Order
    temp_order = np.argsort(5e-1*np.log(1+entropy_arr) + corr_arr)[::-1]
    # temp_order = np.argsort(ks_arr - 5e-2*np.log(entropy_arr))

    # Filter for interest and diversity
    feature_idx = []
    for i in temp_order:
        if len(feature_idx) >= limit:
            break
        if len(feature_idx) == 0:
            feature_idx.append(i)
            continue
        corr = [
            stats.pearsonr(datasets[0][:, i], datasets[0][:, idx])[0]
            for idx in feature_idx]
        corr = [c for c in corr if not np.isnan(c)]
        if all(corr) or len(corr) == 0:
            feature_idx.append(i)

    # By raw score, raw score and diversity
    # print(entropy_arr[temp_order])
    # print(datasets[0][:, feature_idx[0]])
    # print(distribution_true[feature_idx[0]])
    # print(corr_arr[temp_order])
    return temp_order, feature_idx


def hash_kwargs(kwargs, dataset_name, dataset):
    fromChar = [' ', '),', '(', ')', ',', '\'', '[', ']']
    toChar = ['', '--', '', '', '-', '', '(', ')']
    kwargs_str = str(sorted(kwargs.items()))[1:-1]
    for f, t in zip(fromChar, toChar):
        kwargs_str = kwargs_str.replace(f, t)
    size_str = '---'.join([dataset_name, '-'.join([str(s) for s in dataset[0].shape]), '-'.join([str(s) for s in dataset[1].shape])])
    hash_str = '---'.join([size_str, kwargs_str])
    return size_str, hash_str


class preclass:
    def __init__(self, sample, pca=None, axis=None):
        self.sample = sample
        self.pca = pca
        self.axis = axis  # Generally None or 0, depending on if feature magnitude matters

    def transform(self, X):
        out = X
        if self.pca is not None:
            out = self.pca.transform(out)
        out = out - self.sample.mean(self.axis)
        out = out / self.sample.std(self.axis)
        out[np.isnan(out)] = 0
        return out

    def inverse_transform(self, X):
        out = X
        out = out * self.sample.std(self.axis)
        out = out + self.sample.mean(self.axis)
        if self.pca is not None:
            out = self.pca.inverse_transform(out)
        return out


class SimpleJAMIEModel(nn.Module):
    """Small encoder-decoder model"""
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.num_modalities = len(input_dim)
        self.encoders = []
        for i in range(self.num_modalities):
            self.encoders.append(nn.Sequential(
                nn.Linear(input_dim[i], output_dim),
                nn.BatchNorm1d(output_dim),
            ))
        self.encoders = nn.ModuleList(self.encoders)

        self.decoders = []
        for i in range(self.num_modalities):
            self.decoders.append(nn.Sequential(
                nn.Linear(output_dim, input_dim[i]),
                nn.BatchNorm1d(input_dim[i]),
            ))
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, *X, corr=None):
        """Regular forward method"""
        assert corr is not None, '``corr`` must be provided.'
        embedded = [self.encoders[i](X[i]) for i in range(self.num_modalities)]
        combined = [
            (
                embedded[i]
                + torch.mm(
                    corr if i == 0 else torch.t(corr),
                    embedded[(i + 1) % 2])
            ) / (1. + corr.sum((i + 1) % 2).reshape(-1, 1))
            for i in range(self.num_modalities)
        ]
        reconstructed = [self.decoders[i](combined[i]) for i in range(self.num_modalities)]

        return embedded, reconstructed
