import contextlib
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import r_regression
from sklearn.metrics import silhouette_samples, silhouette_score
import torch

from .commando import ComManDo
from .utilities import ensure_list, predict_nn, SimpleDualEncoder


def test_partial(
    datasets,
    types,
    fraction_range=np.linspace(0, 1, 3),
    plot=True,
    **kwargs,
):
    """Takes aligned datasets and tests accuracy based on alignment assumptions"""
    # asddf: Add option to mute output
    assert len(datasets[0]) == len(datasets[1]), 'Datasets must be aligned.'

    types = [np.unique(type, return_inverse=True)[1] for type in types]
    num_samples = len(datasets[0])
    acc_list = {'lta': [], 'foscttm': []}
    for fraction in fraction_range:
        random_idx = np.random.choice(
            range(num_samples), int(fraction * num_samples),
            replace=False,
        )
        random_diag = np.zeros(num_samples)
        random_diag[random_idx] = 1
        cm = ComManDo(P=np.diag(random_diag), **kwargs)
        with contextlib.redirect_stdout(None):
            cm_data = cm.fit_transform(dataset=datasets)
            acc_list['lta'].append(cm.test_LabelTA(cm_data, types))
            acc_list['foscttm'].append(cm.test_closer(cm_data))

    # https://stackoverflow.com/a/42466319
    if plot:
        for k, v in acc_list.items():
            plt.plot(fraction_range, v, '.-', label=k)
        plt.xlabel("Fraction Assumed Aligned")
        plt.ylabel("Statistic")
        plt.legend()
    return acc_list, fraction_range


class generate_figure():
    def __init__(
        self,
        # Data
        cm_trained,
        dataset,
        labels,
        integrated_data=[],
        integrated_alg_names=[],
        alg_groups=None,
        dataset_names=None,
        # Style
        scale=20,
        size_bound=[.25, .5],
        vertical_scale=.75,
        # Visualizations
        reconstruction_features={},
        integrated_use_pca=False,
        raw_data_group=0,
        integrated_rows=3,
        # Simulations
        simple_num_features=32,
        # Remove Visualizations
        exclude_predict=[],
        skip_partial=True,
        skip_nn=True,
        skip_simple=False,
        use_raw_in_integrated=True,
    ):
        """Compares ComManDo with ``integrated_data``"""
        # asddf: Legends may be wrong unless sorted
        assert len(integrated_data) == len(integrated_alg_names), '``alg_*`` params must correspond.'

        # Style
        plt.rcParams.update({
            'figure.titlesize': 20,
            'axes.titlesize': 'x-large',
            'xtick.labelsize': 20,
            'ytick.labelsize': 40,
            'font.size': 40,
            'font.weight': 'bold',
            'font.family': 'normal',
        })
        sns.set(style='darkgrid')

        # Save vars
        # Data
        self.cm_trained = cm_trained
        self.dataset = dataset
        self.labels = labels
        self.integrated_data = ensure_list(integrated_data)
        self.integrated_alg_names = ensure_list(integrated_alg_names)
        self.alg_groups = (
            alg_groups if alg_groups is not None else [0]*len(self.integrated_data))
        self.dataset_names = dataset_names
        # Style
        self.scale = scale
        self.size_bound = size_bound
        self.vertical_scale = vertical_scale
        # Visualizations
        self.reconstruction_features = reconstruction_features
        self.integrated_use_pca = integrated_use_pca
        self.raw_data_group = raw_data_group
        self.integrated_rows = integrated_rows
        # Simulations
        self.simple_num_features = simple_num_features
        # Remove Visualizations
        self.exclude_predict = exclude_predict
        self.skip_partial = skip_partial
        self.skip_nn = skip_nn
        self.skip_simple = skip_simple
        self.use_raw_in_integrated = use_raw_in_integrated

        # Setup
        if self.use_raw_in_integrated:
            concat = [self.raw_data_group, *self.alg_groups]
        else:
            concat = self.alg_groups
        self.unique_groups, self.group_counts = (
            np.unique(concat, return_counts=True))
        self.num_groups = len(self.unique_groups)

        self.types = [np.unique(type, return_inverse=True)[1] for type in self.labels]
        self.num_modalities = len(self.dataset)
        self.num_algorithms = len(self.integrated_data)
        self.dataset_names = [
            dataset_names[i] if dataset_names is not None else f'Dataset {i}'
            for i in range(self.num_modalities)
        ]

        # Sizing
        height_manual_scale = []
        height_numerators = []
        height_denominators = []
        # # Raw Data
        # height_numerators.append(1)
        # height_denominators.append(self.num_modalities)
        # Integrated Data
        height_manual_scale.append(1)
        height_numerators.append(math.ceil((self.num_algorithms + self.use_raw_in_integrated) / 3))
        height_denominators.append(
            min(self.num_algorithms + self.use_raw_in_integrated, 3) * self.num_modalities)
        # Accuracy by Partial
        height_manual_scale.append(.5)
        height_numerators.append(1)
        height_denominators.append(2 - self.skip_partial)
        # Silhouette Value Boxplots
        height_manual_scale.append(.5)
        height_numerators.append(1)
        height_denominators.append(self.num_modalities)
        # Reconstruct Modality
        height_manual_scale.append(1)
        height_numerators.append((self.num_modalities**2 - self.num_modalities) - len(self.exclude_predict))
        height_denominators.append(4 + (not self.skip_nn) + (not self.skip_simple))
        # Ratios
        height_denominators = [
            min(1/size_bound[0], max(1/size_bound[1], x)) for x in height_denominators]
        height_ratios = [
            s*n/d for s, n, d in zip(height_manual_scale, height_numerators, height_denominators)]
        figsize = (scale, scale * vertical_scale * sum(height_ratios))

        # Create figure
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        fig.suptitle(' ')  # To prevent cutoff
        # fig.suptitle('JAMIE Performance')
        subfigs = ensure_list(
            fig.subfigures(len(height_ratios), 1, height_ratios=height_ratios, wspace=.07))
        subfig_idx = 0
        # gridspec = subfigs[0]._subplotspec.get_gridspec()

        # Integrated Data
        cfig = subfigs[subfig_idx]
        csubfigs = ensure_list(cfig.subfigures(1, self.num_groups))
        for i, group in enumerate(self.unique_groups):
            self._plot_integrated_data(csubfigs[i], group_filter=group)
        subfig_idx += 1

        # Accuracy by Partial
        cfig = subfigs[subfig_idx]
        self._plot_accuracy_metrics(cfig)
        subfig_idx += 1

        # Silhouette Value Boxplots
        cfig = subfigs[subfig_idx]
        self._plot_silhouette_value_boxplots(cfig)
        subfig_idx += 1

        # Reconstruct Modality
        cfig = subfigs[subfig_idx]
        self._plot_reconstruct_modality(cfig)
        subfig_idx += 1

    def _plot_raw_data(self, cfig):
        num_modalities = self.num_modalities
        dataset = self.dataset
        labels = self.labels
        dataset_names = self.dataset_names

        for i in range(num_modalities):
            ax = cfig.add_subplot(1, num_modalities, i+1)  # , projection='3d'
            pca_data = PCA(n_components=2).fit_transform(dataset[i])
            for label in np.unique(np.concatenate(labels)):
                pca_data_subset = np.transpose(pca_data[labels[i] == label])
                ax.scatter(*pca_data_subset, s=5., label=label)
                ax.set_aspect('equal', adjustable='box')
            title = dataset_names[i]
            ax.set_title(title)
            ax.set_xlabel('PCA-1')
            ax.set_ylabel('PCA-2')
            # ax.set_zlabel('PCA-3')
        ax.legend()
        # fig.suptitle('Raw Data')

    def _plot_integrated_data(self, cfig, group_filter=None):
        num_algorithms = (
            sum(self.alg_groups == group_filter)
            if group_filter is not None
            else self.num_algorithms
        )
        use_raw_in_integrated = (
            self.use_raw_in_integrated and group_filter == self.raw_data_group)
        dataset = self.dataset
        labels = self.labels
        dataset_names = self.dataset_names
        num_modalities = self.num_modalities
        integrated_data = (
            self.integrated_data[self.alg_groups == group_filter]
            if group_filter is not None
            else self.integrated_data
        )
        integrated_use_pca = self.integrated_use_pca
        integrated_alg_names = (
            self.integrated_alg_names[self.alg_groups == group_filter]
            if group_filter is not None
            else self.integrated_alg_names
        )
        integrated_rows = self.integrated_rows

        csubfigs = cfig.subfigures(
            math.ceil((num_algorithms + use_raw_in_integrated) / integrated_rows),
            min(num_algorithms + use_raw_in_integrated, integrated_rows),
            wspace=.07,
        ).flatten()
        for csubfig, j in zip(csubfigs, range(num_algorithms + use_raw_in_integrated)):
            for i in range(num_modalities):
                if use_raw_in_integrated and j == 0:
                    ax = csubfig.add_subplot(1, num_modalities, i+1)
                    plot_data = PCA(n_components=2).fit_transform(dataset[i])
                    for label in np.unique(np.concatenate(labels)):
                        data_subset = np.transpose(plot_data[labels[i] == label])[:2, :]
                        ax.scatter(*data_subset, s=5., label=label)
                    title = dataset_names[i]
                    ax.set_title(title)
                    type_text = 'PCA'
                    ax.set_xlabel(type_text + '-1')
                    ax.set_ylabel(type_text + '-2')
                    if i == 0:
                        ax.legend()
                    suptitle = 'Raw Data'
                    continue

                ax = csubfig.add_subplot(1, num_modalities, i+1)
                # projection='3d'
                plot_data = integrated_data[j-use_raw_in_integrated][i]
                if integrated_use_pca:
                    plot_data = PCA(n_components=2).fit_transform(plot_data)
                for label in np.unique(np.concatenate(labels)):
                    data_subset = np.transpose(plot_data[labels[i] == label])[:2, :]
                    ax.scatter(*data_subset, s=5., label=label)
                title = dataset_names[i]
                ax.set_title(title)
                if integrated_use_pca:
                    type_text = 'PCA'
                else:
                    type_text = 'Latent Feature'
                ax.set_xlabel(type_text + '-1')
                ax.set_ylabel(type_text + '-2')
                # ax.set_zlabel('Latent Feature 3')
                suptitle = integrated_alg_names[j-use_raw_in_integrated]
            csubfig.suptitle(suptitle)
        # cfig.suptitle('Integrated Embeddings')

    def _plot_accuracy_metrics(self, cfig):
        skip_partial = self.skip_partial
        dataset = self.dataset
        types = self.types
        integrated_alg_names = self.integrated_alg_names
        dataset_names = self.dataset_names
        num_algorithms = self.num_algorithms
        cm_trained = self.cm_trained
        integrated_data = self.integrated_data

        csubfigs = ensure_list(
            cfig.subfigures(1, 2 if not skip_partial else 1, wspace=.07))
        csubfig_idx = 0
        if not skip_partial:
            ax = csubfigs[csubfig_idx].subplots(1, 1)
            acc_list, fraction_range = test_partial(dataset, types, plot=False)
            for k, v in acc_list.items():
                ax.plot(fraction_range, v, '.-', label=k)
            ax.set_title('Accuracy by Partial Alignment')
            ax.set_xlabel('Fraction Assumed Aligned')
            ax.set_ylabel('Statistic')
            ax.legend()
            csubfig_idx += 1

        # Metric by Algorithm
        ax = csubfigs[csubfig_idx].subplots(1, 1)
        acc_dict = {
            'Algorithm': integrated_alg_names,
            'Label Transfer Accuracy': [],
            'FOSCTTM': [],
        }
        for name in dataset_names:
            acc_dict['Silhouette Score:\n' + name] = []
        for i in range(num_algorithms):
            with contextlib.redirect_stdout(None):
                acc_dict['Label Transfer Accuracy'].append(
                    cm_trained.test_LabelTA(integrated_data[i], types))
                acc_dict['FOSCTTM'].append(
                    cm_trained.test_closer(integrated_data[i]))
                for j, name in enumerate(dataset_names):
                    acc_dict['Silhouette Score:\n' + name].append(
                        silhouette_score(integrated_data[i][j], types[j]))
        df = pd.DataFrame(acc_dict).melt(
            id_vars=list(acc_dict.keys())[:1],
            value_vars=list(acc_dict.keys())[1:])
        sns.barplot(data=df, x='variable', y='value', hue='Algorithm', ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title('Metric by Algorithm')
        # cfig.suptitle('Miscellaneous Accuracy Statistics')
        csubfig_idx += 1

    def _plot_distance_by_cell(self, cfig):
        num_algorithms = self.num_algorithms
        integrated_data = self.integrated_data
        labels = self.labels
        integrated_alg_names = self.integrated_alg_names

        axs = cfig.subplots(1, num_algorithms)
        for ax, i in zip(axs, range(num_algorithms)):
            lab, dat = cm_trained.test_label_dist(integrated_data[i], labels, verbose=False)
            # Sort to look nice
            idx = np.argsort(dat, axis=1)[0]
            dat = dat[idx, :][:, idx]
            lab = lab[idx]
            ax = sns.heatmap(dat, xticklabels=lab, yticklabels=lab, linewidth=0, cmap='YlGnBu', ax=ax)
            ax.set_title(integrated_alg_names[i])
        cfig.suptitle('Distance of Medoid by Cell Type')

    def _plot_silhouette_value_boxplots(self, cfig):
        num_modalities = self.num_modalities
        num_algorithms = self.num_algorithms
        integrated_data = self.integrated_data
        types = self.types
        labels = self.labels
        integrated_alg_names = self.integrated_alg_names
        dataset_names = self.dataset_names

        axs = cfig.subplots(1, num_modalities)
        for i, ax in enumerate(axs):
            # Calculate coefficients
            df = pd.DataFrame(columns=['Algorithm', 'Cell', 'Silhouette Coefficient'])
            for j in range(num_algorithms):
                coefs = silhouette_samples(integrated_data[j][i], types[i])
                for label in np.unique(np.concatenate(labels)):
                    for value in coefs[labels[i] == label]:
                        df = df.append({
                            'Algorithm': integrated_alg_names[j],
                            'Cell': label,
                            'Silhouette Coefficient': value,
                        }, ignore_index=True)

            # Plot
            sns.boxplot(
                data=df,
                x='Cell',
                y='Silhouette Coefficient',
                hue='Algorithm',
                ax=ax,
            )
            ax.set_title(dataset_names[i])
            ax.legend([], [], frameon=False)
        # cfig.suptitle('Silhouette Score by Cell Type')

    def _plot_reconstruct_modality(self, cfig):
        """Plot a module to assess modal translation efficacy"""
        num_modalities = self.num_modalities
        exclude_predict = self.exclude_predict
        skip_nn = self.skip_nn
        skip_simple = self.skip_simple
        cm_trained = self.cm_trained
        dataset = self.dataset
        reconstruction_features = self.reconstruction_features
        labels = self.labels
        dataset_names = self.dataset_names
        simple_num_features = self.simple_num_features

        csubfigs = ensure_list(cfig.subfigures(
            (num_modalities**2 - num_modalities) - len(exclude_predict),
            1,
            wspace=.07
        ))
        fig_idx = 0
        for i in range(num_modalities):
            for j in range(num_modalities):
                if i == j or ((i, j) in exclude_predict):
                    continue
                csubfig = csubfigs[fig_idx]
                axs = csubfig.subplots(1, 4 + (not skip_nn) + (not skip_simple))
                fig_idx += 1

                # csubfig.suptitle(f'{dataset_names[i]} -> {dataset_names[j]}')

                predicted = cm_trained.modal_predict(dataset[i], i)
                actual = dataset[j]

                # Setup
                if (i, j) in reconstruction_features:
                    feat = reconstruction_features[(i, j)]
                else:
                    feat = [0, 1]

                axi = 0
                # Real
                for label in np.unique(np.concatenate(labels)):
                    subdata = np.transpose(actual[:, feat][labels[j] == label])
                    axs[axi].scatter(*subdata, label=label, s=5.)
                axs[axi].set_title(f'Actual {dataset_names[j]}')
                axs[axi].set_xlabel('Latent Feature 1')
                axs[axi].set_ylabel('Latent Feature 2')
                axi += 1

                # Predicted
                for label in np.unique(np.concatenate(labels)):
                    subdata = np.transpose(predicted[:, feat][labels[j] == label])
                    axs[axi].scatter(*subdata, label=label, s=5.)
                axs[axi].set_title('ComManDo Predicted')
                axs[axi].set_xlabel('Latent Feature 1')
                axs[axi].set_ylabel('Latent Feature 2')
                axi += 1

                # NN Predicted
                if not skip_nn:
                    nn_predicted = predict_nn(
                        torch.tensor(dataset[i]).float(), torch.tensor(dataset[j]).float())
                    for label in np.unique(np.concatenate(labels)):
                        subdata = np.transpose(nn_predicted[:, feat][labels[j] == label])
                        axs[axi].scatter(*subdata, label=label, s=5.)
                    axs[axi].set_title('NN Predicted')
                    axs[axi].set_xlabel('Latent Feature 1')
                    axs[axi].set_ylabel('Latent Feature 2')
                    axi += 1

                # Correlation
                corr = []
                for k in range(predicted.shape[1]):
                    corr.append(r_regression(predicted[:, [k]], actual[:, k])[0])
                axs[axi].bar(range(len(corr)), corr)
                axs[axi].set_title('Correlation by Feature')
                axs[axi].set_xlabel('Feature')
                axs[axi].set_ylabel('Correlation')
                axi += 1

                # Cherry-Picked Prediction
                for label in np.unique(np.concatenate(labels)):
                    true = np.transpose(actual[:, feat[0]][labels[j] == label])
                    pred = np.transpose(predicted[:, feat[0]][labels[j] == label])
                    axs[axi].scatter(true, pred, label=label, s=5.)
                axs[axi].set_title('Predicted vs Truth')
                axs[axi].set_xlabel('True Value')
                axs[axi].set_ylabel('Predicted Value')
                axi += 1

                # Simple model
                if not skip_simple:
                    simple_cm = ComManDo(
                        model_class=SimpleDualEncoder, output_dim=simple_num_features)
                    with contextlib.redirect_stdout(None):
                        simple_cm.fit_transform(dataset=dataset)
                    average_weights = (
                        simple_cm.model.encoders[0][0].weight.detach().cpu().numpy()
                        + np.transpose(simple_cm.model.decoders[0][0].weight.detach().cpu().numpy())
                    ).sum(axis=0) / 2
                    axs[axi].bar(range(len(average_weights)), average_weights)
                    axs[axi].set_title('Weight by Feature')
                    axs[axi].set_xlabel('Feature')
                    axs[axi].set_ylabel('Linear Encoding Weight')
                    axi += 1
        # cfig.suptitle('Modality Prediction')

        # plt.tight_layout()
        # plt.subplots_adjust(top=1)
        plt.show()
