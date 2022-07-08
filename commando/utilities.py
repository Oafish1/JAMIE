import contextlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import umap


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
    def __init__(self, input_dim, output_dim, hidden_dim=1, p=0.6):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass for the model"""
        return self.fc2(self.dropout(self.fc1(x)))


class SingleModel(nn.Module):
    """Thin, single-layer NN model"""
    def __init__(self, input_dim, output_dim, p=0.6):
        super().__init__()

        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """Forward pass for the model"""
        return self.fc1(self.dropout(x))


def predict_nn(source, target, val=None, epochs=200, batches=10):
    """Predict modality using a simple NN"""
    model = SimpleModel(source.shape[1], target.shape[1])
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.MSELoss()
    batch_size = int(len(source)/batches)

    for epoch in range(epochs):
        for _ in range(batches):
            idx = np.random.choice(range(len(source)), batch_size, replace=False)
            optimizer.zero_grad()
            logits = model(source[idx])
            loss = criterion(logits, target[idx])
            loss.backward()
            optimizer.step()
    if val is not None:
        return model(val).detach().cpu().numpy()
    return model(source).detach().cpu().numpy()


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


class SimpleDualEncoder(nn.Module):
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
