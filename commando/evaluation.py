import contextlib
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import r_regression
from sklearn.metrics import davies_bouldin_score, silhouette_samples
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
    """
    Class for visualizing results of multiple algorithms, performing embedding,
    correspondence analysis, and modal translation.
    """
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
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
        heat_cmap=[sns.cm.mako, sns.cm.mako_r],
        heatmap_soften_factor=1,
        # Visualizations
        reconstruction_features={},
        integrated_use_pca=False,
        raw_data_group=0,
        integrated_rows=1,
        # Simulations
        simple_num_features=32,
        num_best_reconstructed_features=3,
        # Remove Visualizations
        exclude_predict=[],
        skip_partial=True,
        skip_nn=True,
        skip_simple=True,
        use_raw_in_integrated=True,
    ):
        # asddf: Legends may be wrong unless sorted
        assert len(integrated_data) == len(integrated_alg_names), (
            '``alg_*`` params must correspond.')

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
        self.heat_cmap = heat_cmap
        self.heatmap_soften_factor = heatmap_soften_factor
        # Visualizations
        self.reconstruction_features = reconstruction_features
        self.integrated_use_pca = integrated_use_pca
        self.raw_data_group = raw_data_group
        self.integrated_rows = integrated_rows
        # Simulations
        self.simple_num_features = simple_num_features
        self.num_best_reconstructed_features = num_best_reconstructed_features
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

        self.colors = np.array(colors[:self.num_algorithms])

        # What to plot
        to_run = (
            lambda x: self._group_plot(x, self._plot_integrated_data),
            self._plot_accuracy_metrics,
            self._plot_silhouette_value_boxplots,
            self._plot_reconstruct_modality,
        )
        to_run_size = (
            self._group_shape(self._get_integrated_data_shape()),
            self._get_accuracy_metrics_shape(),
            self._get_silhouette_value_boxplots_shape(),
            self._get_reconstruct_modality_shape(),
        )

        # Sizing
        height_manual_scale = [shape[0] for shape in to_run_size]
        height_numerators = [shape[1] for shape in to_run_size]
        height_denominators = [shape[2] for shape in to_run_size]
        height_denominators = [
            min(1/size_bound[0], max(1/size_bound[1], x)) for x in height_denominators]
        height_ratios = [
            s*n/d for s, n, d in zip(height_manual_scale, height_numerators, height_denominators)]
        figsize = (scale, scale * vertical_scale * sum(height_ratios))

        # Create figure
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        fig.suptitle(' ')
        subfigs = ensure_list(
            fig.subfigures(len(height_ratios), 1, height_ratios=height_ratios, wspace=.07))
        # gridspec = subfigs[0]._subplotspec.get_gridspec()

        # Plot
        assert len(subfigs) == len(to_run), '``to_run`` and ``to_run_size`` must match in shape'
        for cfig, to_run_func in zip(subfigs, to_run):
            to_run_func(cfig)

    def _group_shape(self, shape):
        shape_array = [x for x in shape]
        shape_array[-1] *= self.num_groups
        return shape_array

    def _group_plot(self, cfig, func):
        csubfigs = ensure_list(cfig.subfigures(1, self.num_groups))
        for i, group in enumerate(self.unique_groups):
            func(csubfigs[i], group_filter=group)

    def _get_integrated_group(self, group_filter=None):
        num_algorithms = (
            sum(self.alg_groups == group_filter)
            if group_filter is not None
            else self.num_algorithms
        )
        integrated_data = (
            self.integrated_data[self.alg_groups == group_filter]
            if group_filter is not None
            else self.integrated_data
        )
        integrated_alg_names = (
            self.integrated_alg_names[self.alg_groups == group_filter]
            if group_filter is not None
            else self.integrated_alg_names
        )
        colors = (
            self.colors[self.alg_groups == group_filter]
            if group_filter is not None
            else self.colors
        )
        use_raw_in_integrated = (
            self.use_raw_in_integrated and group_filter == self.raw_data_group)
        return (
            num_algorithms,
            integrated_data,
            integrated_alg_names,
            colors,
            use_raw_in_integrated,
        )

    def _get_raw_data_shape(self):
        scale = 1
        rows = 1
        cols = self.num_modalities
        return scale, rows, cols

    def _plot_raw_data(self, cfig):
        num_modalities = self.num_modalities
        dataset = self.dataset
        labels = self.labels
        dataset_names = self.dataset_names

        for i in range(num_modalities):
            ax = cfig.add_subplot(1, num_modalities, i+1)
            pca_data = PCA(n_components=2).fit_transform(dataset[i])
            for label in np.unique(np.concatenate(labels)):
                pca_data_subset = np.transpose(pca_data[labels[i] == label])
                ax.scatter(*pca_data_subset, s=5., label=label)
            title = dataset_names[i]
            ax.set_title(title)
            ax.set_xlabel('PCA-1')
            ax.set_ylabel('PCA-2')
            ax.set_aspect('equal', adjustable='box')
        ax.legend()
        # fig.suptitle('Raw Data')

    def _get_integrated_data_shape(self):
        scale = 1
        rows = math.ceil(max(self.group_counts) / self.integrated_rows)
        cols = self.num_modalities * self.integrated_rows
        return scale, rows, cols

    def _plot_integrated_data(self, cfig, group_filter=None):
        dataset = self.dataset
        labels = self.labels
        dataset_names = self.dataset_names
        num_modalities = self.num_modalities
        integrated_use_pca = self.integrated_use_pca
        integrated_rows = self.integrated_rows
        num_algorithms, integrated_data, integrated_alg_names, _, use_raw_in_integrated = (
            self._get_integrated_group(group_filter))

        csubfigs = cfig.subfigures(
            math.ceil((num_algorithms + use_raw_in_integrated) / integrated_rows),
            min(num_algorithms + use_raw_in_integrated, integrated_rows),
            wspace=.07,
        ).flatten()
        for csubfig, j in zip(csubfigs, range(num_algorithms + use_raw_in_integrated)):
            for i in range(num_modalities):
                if use_raw_in_integrated and j == 0:
                    ax = csubfig.add_subplot(1, num_modalities, i+1)  # , projection='3d'
                    plot_data = PCA(n_components=2).fit_transform(dataset[i])
                    for label in np.unique(np.concatenate(labels)):
                        data_subset = np.transpose(plot_data[labels[i] == label])[:2, :]
                        ax.scatter(*data_subset, s=5., label=label)
                    title = dataset_names[i]
                    ax.set_title(title)
                    type_text = 'PC'
                    ax.set_xlabel(type_text + '-1')
                    ax.set_ylabel(type_text + '-2')
                    # ax.set_aspect('equal', adjustable='box')
                    if i == 0:
                        ax.legend()
                    suptitle = 'Raw Data'
                    continue

                ax = csubfig.add_subplot(1, num_modalities, i+1)
                # projection='3d'
                plot_data = integrated_data[j-use_raw_in_integrated][i]
                if integrated_use_pca:
                    if i == 0:
                        pca = PCA(n_components=2)
                        pca.fit(plot_data)
                    plot_data = pca.transform(plot_data)
                for label in np.unique(np.concatenate(labels)):
                    data_subset = np.transpose(plot_data[labels[i] == label])[:2, :]
                    ax.scatter(*data_subset, s=5., label=label)
                title = dataset_names[i]
                ax.set_title(title)
                if integrated_use_pca:
                    type_text = 'PC'
                else:
                    type_text = 'Feature'
                ax.set_xlabel(type_text + '-1')
                ax.set_ylabel(type_text + '-2')
                # ax.set_aspect('equal', adjustable='box')
                # ax.set_zlabel('Latent Feature 3')
                suptitle = integrated_alg_names[j-use_raw_in_integrated]
            csubfig.suptitle(suptitle)
        # cfig.suptitle('Integrated Embeddings')

    def _get_accuracy_metrics_shape(self):
        scale = .75
        rows = 1
        cols = (2 - self.skip_partial)
        return scale, rows, cols

    def _plot_accuracy_metrics(self, cfig, group_filter=None):
        skip_partial = self.skip_partial
        dataset = self.dataset
        types = self.types
        dataset_names = self.dataset_names
        cm_trained = self.cm_trained
        num_algorithms, integrated_data, integrated_alg_names, colors, _ = (
            self._get_integrated_group(group_filter))

        csubfigs = ensure_list(
            cfig.subfigures(1, 3 if not skip_partial else 2, wspace=.07))
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
        acc_dict = {
            'Algorithm': integrated_alg_names,
            'Label Transfer Accuracy': [],
            'FOSCTTM': [],
        }
        for name in dataset_names:
            acc_dict['Davies-Bouldin:\n' + name] = []
        for i in range(num_algorithms):
            with contextlib.redirect_stdout(None):
                acc_dict['Label Transfer Accuracy'].append(
                    cm_trained.test_LabelTA(integrated_data[i], types))
                acc_dict['FOSCTTM'].append(
                    cm_trained.test_closer(integrated_data[i]))
                for j, name in enumerate(dataset_names):
                    acc_dict['Davies-Bouldin:\n' + name].append(
                        davies_bouldin_score(integrated_data[i][j], types[j]))
        keys_01 = ['Algorithm', 'Label Transfer Accuracy', 'FOSCTTM']
        keys_0i = ['Algorithm'] + ['Davies-Bouldin:\n' + name for name in dataset_names]
        df_01, df_0i = (pd.DataFrame({k: v for k, v in acc_dict.items() if k in keys}).melt(
            id_vars=list(keys)[:1],
            value_vars=list(keys)[1:])
            for keys in (keys_01, keys_0i))
        dfs = [df_01, df_0i]
        for i, df in enumerate(dfs):
            ax = csubfigs[csubfig_idx].subplots(1, 1)
            pl = sns.barplot(
                data=df,
                x='variable',
                y='value',
                hue='Algorithm',
                ax=ax,
                palette=colors)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title('Metrics by Algorithm')
            if i!=0:
                pl.legend_.remove()
            csubfig_idx += 1
        # cfig.suptitle('Miscellaneous Accuracy Statistics')

    def _get_accuracy_metrics_heatmap_shape(self):
        scale = .75
        rows = 1
        cols = 1
        return scale, rows, cols

    def _plot_accuracy_metrics_heatmap(self, cfig, group_filter=None):
        types = self.types
        dataset_names = self.dataset_names
        cm_trained = self.cm_trained
        num_algorithms, integrated_data, integrated_alg_names, colors, _ = (
            self._get_integrated_group(group_filter))

        # Metric by Algorithm
        reverse_color = [False, True] + [True] * self.num_modalities  # True iff lower is better
        acc_dict = {
            'Algorithm': integrated_alg_names,
            'Label Transfer Accuracy': [],
            'FOSCTTM': [],
        }
        for name in dataset_names:
            acc_dict['Davies-Bouldin:\n' + name] = []
        for i in range(num_algorithms):
            with contextlib.redirect_stdout(None):
                acc_dict['Label Transfer Accuracy'].append(
                    cm_trained.test_LabelTA(integrated_data[i], types))
                acc_dict['FOSCTTM'].append(
                    cm_trained.test_closer(integrated_data[i]))
                for j, name in enumerate(dataset_names):
                    acc_dict['Davies-Bouldin:\n' + name].append(
                        davies_bouldin_score(integrated_data[i][j], types[j]))
        df = pd.DataFrame(acc_dict)
        df = df.set_index('Algorithm')
        axs = cfig.subplots(1, len(df), gridspec_kw={'wspace': 0})
        for i, (ax, col) in enumerate(zip(axs, df.columns)):
            # Avoid harsh black
            sf = self.heatmap_soften_factor
            if not reverse_color[i]:
                vmin = df[col].values.min() - sf * (df[col].values.max() - df[col].values.min())
                vmax = df[col].values.max()
            else:
                vmin = df[col].values.min()
                vmax = df[col].values.max() + sf * (df[col].values.max() - df[col].values.min())
            # Plot heatmap
            sns.heatmap(
                np.array([df[col].values]).T,
                xticklabels=[col],
                yticklabels=df.index,
                ax=ax,
                annot=True,
                fmt='.2f',
                cbar=False,
                vmin=vmin,
                vmax=vmax,
                cmap=self.heat_cmap[0] if not reverse_color[i] else self.heat_cmap[1])
            if i > 0:
                ax.yaxis.set_ticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title('Metric by Algorithm')

    def _plot_distance_by_cell(self, cfig, group_filter=None):
        cm_trained = self.cm_trained
        num_algorithms = self.num_algorithms
        labels = self.labels
        _, integrated_data, integrated_alg_names, _, _ = (
            self._get_integrated_group(group_filter))

        axs = cfig.subplots(1, num_algorithms)
        for ax, i in zip(axs, range(num_algorithms)):
            lab, dat = cm_trained.test_label_dist(integrated_data[i], labels, verbose=False)
            # Sort to look nice
            idx = np.argsort(dat, axis=1)[0]
            dat = dat[idx, :][:, idx]
            lab = lab[idx]
            ax = sns.heatmap(
                dat,
                xticklabels=lab,
                yticklabels=lab,
                linewidth=0,
                cmap='YlGnBu',
                ax=ax
            )
            ax.set_title(integrated_alg_names[i])
        cfig.suptitle('Distance of Medoid by Cell Type')

    def _get_silhouette_value_boxplots_shape(self):
        scale = .5
        rows = 1
        cols = self.num_modalities
        return scale, rows, cols

    def _plot_silhouette_value_boxplots(self, cfig, group_filter=None):
        num_modalities = self.num_modalities
        types = self.types
        labels = self.labels
        dataset_names = self.dataset_names
        num_algorithms, integrated_data, integrated_alg_names, colors, _ = (
            self._get_integrated_group(group_filter))

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
                palette=colors,
            )
            ax.set_title(dataset_names[i])
            if i == 0:
                ax.legend()
            else:
                ax.legend([], [], frameon=False)
        # cfig.suptitle('Silhouette Score by Cell Type')

    def _get_reconstruct_modality_shape(self):
        scale = 1
        rows = (
            (self.num_modalities**2 - self.num_modalities) - len(self.exclude_predict)
        ) * 2
        cols = 4 + (not self.skip_nn) + (not self.skip_simple)
        return scale, rows, cols

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

        _, rows, cols = self._get_reconstruct_modality_shape()
        csubfigs = ensure_list(cfig.subfigures(rows, 1, wspace=.07))
        fig_idx = 0
        for i in range(num_modalities):
            for j in range(num_modalities):
                if i == j or ((i, j) in exclude_predict):
                    continue
                csubfig = csubfigs[fig_idx]
                axs = csubfig.subplots(1, 4 + (not skip_nn) + (not skip_simple))
                # csubfig.suptitle(f'{dataset_names[i]} -> {dataset_names[j]}')
                fig_idx += 1

                # Perform prediction (after preprocessing, if applicable)
                predicted = cm_trained.modal_predict(dataset[i], i)
                actual = cm_trained.model.preprocessing[j](dataset[j])

                # Get correlations per feature
                corr_per_feature = []
                for k in range(predicted.shape[1]):
                    corr_per_feature.append(r_regression(predicted[:, [k]], actual[:, k])[0])
                # Get best features
                sorted_feature_idx = np.argsort(
                    np.nan_to_num(corr_per_feature))[::-1]
                feat = sorted_feature_idx[:self.num_best_reconstructed_features]
                # Sort
                corr_per_feature.sort()

                # Get correlations per sample
                corr_per_sample = []
                for k in range(predicted.shape[0]):
                    corr_per_sample.append(r_regression(
                        np.transpose(predicted[[k], :]), np.transpose(actual[k, :]))[0])
                # Sort
                corr_per_sample.sort()

                # Setup
                if (i, j) in reconstruction_features:
                    feat = reconstruction_features[(i, j)]

                axi = 0
                # Real
                for label in np.unique(np.concatenate(labels)):
                    subdata = np.transpose(actual[:, feat[:2]][labels[j] == label])
                    axs[axi].scatter(*subdata, label=label, s=5.)
                axs[axi].set_title(f'Actual {dataset_names[j]}')
                axs[axi].set_xlabel('Latent Feature 1')
                axs[axi].set_ylabel('Latent Feature 2')
                axi += 1

                # Predicted
                for label in np.unique(np.concatenate(labels)):
                    subdata = np.transpose(predicted[:, feat[:2]][labels[j] == label])
                    axs[axi].scatter(*subdata, label=label, s=5.)
                axs[axi].set_title(f'JAMIE Translated {dataset_names[j]}')
                axs[axi].set_xlabel('Latent Feature 1')
                axs[axi].set_ylabel('Latent Feature 2')
                axi += 1

                # NN Predicted
                if not skip_nn:
                    nn_predicted = predict_nn(
                        torch.tensor(dataset[i]).float(), torch.tensor(dataset[j]).float())
                    for label in np.unique(np.concatenate(labels)):
                        subdata = np.transpose(nn_predicted[:, feat[:2]][labels[j] == label])
                        axs[axi].scatter(*subdata, label=label, s=5.)
                    axs[axi].set_title('NN Predicted')
                    axs[axi].set_xlabel('Latent Feature 1')
                    axs[axi].set_ylabel('Latent Feature 2')
                    axi += 1

                # Correlation per feature
                axs[axi].plot(np.linspace(0, 1, len(corr_per_feature)), corr_per_feature)
                axs[axi].set_title('Prediction Correlation by Feature')
                axs[axi].set_xlabel('Percentile of Features')
                axs[axi].set_ylabel('Correlation')
                axi += 1

                # Correlation per sample
                axs[axi].plot(np.linspace(0, 1, len(corr_per_sample)), corr_per_sample)
                axs[axi].set_title('Prediction Correlation by Sample')
                axs[axi].set_xlabel('Percentile of Samples')
                axs[axi].set_ylabel('Correlation')
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

                # Predicted vs True section
                csubfig = csubfigs[fig_idx]
                axs = csubfig.subplots(1, len(feat))
                fig_idx += 1
                for feature_idx, ax in zip(feat, axs):
                    for label in np.unique(np.concatenate(labels)):
                        true = np.transpose(actual[:, feature_idx][labels[j] == label])
                        pred = np.transpose(predicted[:, feature_idx][labels[j] == label])
                        ax.scatter(true, pred, label=label, s=5.)
                    ax.set_title(f'Predicted vs True {dataset_names[j]}')
                    ax.set_xlabel('True Value')
                    ax.set_ylabel('Predicted Value')
        # cfig.suptitle('Modality Prediction')

        # plt.tight_layout()
        # plt.subplots_adjust(top=1)
        plt.show()
