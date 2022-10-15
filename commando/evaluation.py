import contextlib
import math

from adjustText import adjust_text
from brokenaxes import brokenaxes
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression, r_regression
from sklearn.metrics import (
    davies_bouldin_score, roc_auc_score, roc_curve, silhouette_samples)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
import torch
import umap

from .commando import ComManDo
from .utilities import (
    ensure_list, jensen_shannon_from_array, outliers, predict_nn,
    set_yticks, sort_by_interest, SimpleJAMIEModel)


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
        integrated_alg_shortnames=None,
        alg_groups=None,
        dataset_names=None,
        feature_names=None,
        # Style
        scale=20,
        dpi=300,
        legend_ncol=2,
        size_bound=[.25, .5],
        vertical_scale=.75,
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
        heat_cmap=[sns.cm.mako, sns.cm.mako_r],
        heatmap_soften_factor=1,
        # Visualizations
        reconstruction_features={},
        integrated_use_pca=False,
        raw_data_group=0,
        raw_legend_plot=0,
        integrated_rows=1,
        # Simulations
        simple_num_features=32,
        num_best_reconstructed_features=4,
        show_sorted_features={},
        # Remove Visualizations
        exclude_predict=[],
        skip_partial=True,
        skip_nn=True,
        skip_simple=True,
        use_raw_in_integrated=False,
    ):
        # asddf: Legends may be wrong unless sorted
        assert len(integrated_data) == len(integrated_alg_names), (
            '``alg_*`` params must correspond.')

        # Style
        sns.set(style='darkgrid')
        plt.rcParams.update({
            'figure.titlesize': 10,
            'axes.titlesize': 20,
            'axes.labelsize': 15,
            'xtick.labelsize': 12.5,
            'ytick.labelsize': 12.5,
            'legend.fontsize': 12.5,
            'font.size': 10,

            # 'axes.titlesize': 'x-large',
            'font.weight': 'bold',
            # 'font.family': 'normal',
        })

        # Save vars
        # Data
        self.cm_trained = cm_trained
        self.dataset = dataset
        self.labels = labels
        self.integrated_data = ensure_list(integrated_data)
        self.integrated_alg_names = ensure_list(integrated_alg_names)
        if integrated_alg_shortnames is None:
            self.integrated_alg_shortnames = self.integrated_alg_names
        else:
            self.integrated_alg_shortnames = ensure_list(integrated_alg_shortnames)
        self.alg_groups = (
            alg_groups if alg_groups is not None else [0]*len(self.integrated_data))
        self.dataset_names = dataset_names
        self.feature_names = feature_names
        # Style
        self.scale = scale
        self.dpi = dpi
        self.legend_ncol = legend_ncol
        self.size_bound = size_bound
        self.vertical_scale = vertical_scale
        self.heat_cmap = heat_cmap
        self.heatmap_soften_factor = heatmap_soften_factor
        # Visualizations
        self.reconstruction_features = reconstruction_features
        self.integrated_use_pca = integrated_use_pca
        self.raw_data_group = raw_data_group
        self.raw_legend_plot = raw_legend_plot
        self.integrated_rows = integrated_rows
        # Simulations
        self.simple_num_features = simple_num_features
        self.num_best_reconstructed_features = num_best_reconstructed_features
        self.show_sorted_features = show_sorted_features
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

    def plot(self, to_run, to_run_size=None):
        """Perform plotting"""
        if to_run_size is None:
            to_run_size = len(to_run) * [(1, 1, 1)]
        # Sizing
        height_manual_scale = [shape[0] for shape in to_run_size]
        height_numerators = [shape[1] for shape in to_run_size]
        height_denominators = [shape[2] for shape in to_run_size]
        height_denominators = [
            min(1/self.size_bound[0], max(1/self.size_bound[1], x)) for x in height_denominators]
        height_ratios = [
            s*n/d for s, n, d in zip(height_manual_scale, height_numerators, height_denominators)]
        figsize = (self.scale, self.scale * self.vertical_scale * sum(height_ratios))

        # Create figure
        self._fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=self.dpi)
        self._fig.suptitle(' ')
        subfigs = ensure_list(
            self._fig.subfigures(len(height_ratios), 1, height_ratios=height_ratios, wspace=.07))
        # gridspec = subfigs[0]._subplotspec.get_gridspec()

        # Plot
        assert len(subfigs) == len(to_run), '``to_run`` and ``to_run_size`` must match in shape'
        for cfig, to_run_func in zip(subfigs, to_run):
            to_run_func(cfig)
        plt.show()

    def get_fig(self):
        """Return figure object, must be run beforehand"""
        return self._fig

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
        integrated_alg_shortnames = (
            self.integrated_alg_shortnames[self.alg_groups == group_filter]
            if group_filter is not None
            else self.integrated_alg_shortnames
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
            integrated_alg_shortnames,
            colors,
            use_raw_in_integrated,
        )

    def _get_raw_data_shape(self):
        scale = .5
        rows = 1
        cols = self.num_modalities
        return scale, rows, cols

    def _plot_raw_data(self, cfig, ps_scale=1):
        num_modalities = self.num_modalities
        dataset = self.dataset
        labels = self.labels
        dataset_names = self.dataset_names

        full_scale = int(1 / ps_scale) * num_modalities

        for i in range(num_modalities):
            ax = cfig.add_subplot(1, full_scale, i+1)
            pca_data = PCA(n_components=2).fit_transform(dataset[i])
            for label in np.unique(np.concatenate(labels)):
                pca_data_subset = np.transpose(pca_data[labels[i] == label])
                ax.scatter(*pca_data_subset, s=5., label=label)
            title = dataset_names[i]
            ax.set_title(title)
            ax.set_xlabel('PC-1')
            ax.set_ylabel('PC-2')
            # ax.set_aspect('equal', adjustable='box')
            if i == self.raw_legend_plot:
                ax.legend(ncol=self.legend_ncol)
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
        (
            num_algorithms,
            integrated_data,
            integrated_alg_names,
            integrated_alg_shortnames,
            _,
            use_raw_in_integrated,
        ) = self._get_integrated_group(group_filter)

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
                        ax.legend(ncol=self.legend_ncol)
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
                suptitle = integrated_alg_shortnames[j-use_raw_in_integrated]
                title = dataset_names[i]
                ax.set_title(suptitle + ' - ' + title)
                if integrated_use_pca:
                    type_text = 'Component'
                else:
                    type_text = 'Feature'
                ax.set_xlabel(type_text + '-1')
                ax.set_ylabel(type_text + '-2')
                # ax.set_aspect('equal', adjustable='box')
                # ax.set_zlabel('Latent Feature 3')

    def _get_accuracy_metrics_shape(self):
        scale = .5
        rows = 1
        cols = (2 - self.skip_partial)
        return scale, rows, cols

    def _plot_accuracy_metrics(self, cfig, group_filter=None):
        skip_partial = self.skip_partial
        dataset = self.dataset
        types = self.types
        dataset_names = self.dataset_names
        cm_trained = self.cm_trained
        num_algorithms, integrated_data, integrated_alg_names, _, colors, _ = (
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
                lta, k = cm_trained.test_LabelTA(integrated_data[i], types, return_k=True)
                acc_dict['Label Transfer Accuracy'].append(lta)
                acc_dict['FOSCTTM'].append(cm_trained.test_closer(integrated_data[i]))
                for j, name in enumerate(dataset_names):
                    acc_dict['Davies-Bouldin:\n' + name].append(
                        davies_bouldin_score(integrated_data[i][j], types[j]))
        acc_dict[f'Label Transfer Accuracy (k={k})'] = acc_dict.pop('Label Transfer Accuracy')
        keys_01 = ['Algorithm', f'Label Transfer Accuracy (k={k})', 'FOSCTTM']
        keys_0i = ['Algorithm'] + ['Davies-Bouldin:\n' + name for name in dataset_names]
        df_01, df_0i = (pd.DataFrame({k: v for k, v in acc_dict.items() if k in keys}).melt(
            id_vars=list(keys)[:1],
            value_vars=list(keys)[1:])
            for keys in (keys_01, keys_0i))
        dfs = [df_01, df_0i]
        df_names = ['Type Prediction Efficacy', 'Worst-Case Cluster Mixing']
        df_log = [False, True]
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
            ax.set_title(df_names[i])
            if df_log[i]:
                pl.set_yscale('log')
                ticks = [2, 5, 10]
                pl.set_yticks(ticks)
                pl.set_yticklabels(ticks)
            if i != 0:
                pl.legend_.remove()
            else:
                plt.legend(ncol=self.legend_ncol)
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
        cfig.suptitle('Distance of Medoid by Type')

    def _get_silhouette_value_boxplots_shape(self):
        scale = .5
        rows = 1
        cols = self.num_modalities
        return scale, rows, cols

    def _plot_silhouette_value_boxplots(self, cfig, group_filter=None, legend=False):
        num_modalities = self.num_modalities
        types = self.types
        labels = self.labels
        dataset_names = self.dataset_names
        num_algorithms, integrated_data, integrated_alg_names, _, colors, _ = (
            self._get_integrated_group(group_filter))

        axs = cfig.subplots(1, num_modalities)
        for i, ax in enumerate(axs):
            # Calculate coefficients
            df = pd.DataFrame(columns=['Algorithm', 'Type', 'Silhouette Coefficient'])
            for j in range(num_algorithms):
                coefs = silhouette_samples(integrated_data[j][i], types[i])
                for label in np.unique(np.concatenate(labels)):
                    for value in coefs[labels[i] == label]:
                        df = df.append({
                            'Algorithm': integrated_alg_names[j],
                            'Type': label,
                            'Silhouette Coefficient': value,
                        }, ignore_index=True)

            # Plot
            sns.boxplot(
                data=df,
                x='Type',
                y='Silhouette Coefficient',
                hue='Algorithm',
                ax=ax,
                palette=colors,
            )
            ax.set_title('Type Separability on ' + dataset_names[i] + ' Latent Space')
            if i == 0 and legend:
                ax.legend()
            else:
                ax.legend([], [], frameon=False)
        # cfig.suptitle('Silhouette Score by Type')

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
        show_sorted_features = self.show_sorted_features
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
                p_per_feature = []
                for k in range(predicted.shape[1]):
                    # There is a more optimal way to do this
                    corr_per_feature.append(r_regression(predicted[:, [k]], actual[:, k])[0])
                    p_per_feature.append(f_regression(predicted[:, [k]], actual[:, k])[1][0])
                corr_per_feature = np.array(corr_per_feature)
                p_per_feature = np.array(p_per_feature)
                # Get best features
                feat = np.argsort(np.nan_to_num(corr_per_feature))[::-1]
                if (i, j) in show_sorted_features:
                    feat_show_idx = self.show_sorted_features[(i, j)]
                else:
                    feat_show_idx = [0, 1]

                # Get correlations per sample
                corr_per_sample = []
                p_per_sample = []
                for k in range(predicted.shape[0]):
                    corr_per_sample.append(r_regression(
                        np.transpose(predicted[[k], :]), np.transpose(actual[k, :]))[0])
                    p_per_sample.append(f_regression(
                        np.transpose(predicted[[k], :]), np.transpose(actual[k, :]))[0])
                corr_per_sample = np.array(corr_per_sample)
                p_per_sample = np.array(p_per_sample)
                samp = np.argsort(np.nan_to_num(corr_per_sample))[::-1]

                # Setup
                if (i, j) in reconstruction_features:
                    feat = reconstruction_features[(i, j)]
                feat_names = [f'Feature {i+1}' for i in range(len(feat))]
                if self.feature_names is not None and self.feature_names[j] is not None:
                    feat_names = self.feature_names[j]
                axi = 0
                # Real
                for label in np.unique(np.concatenate(labels)):
                    subdata = np.transpose(actual[:, feat[feat_show_idx]][labels[j] == label])
                    axs[axi].scatter(*subdata, label=label, s=5.)
                axs[axi].set_title(f'True {dataset_names[j]}')
                axs[axi].set_xlabel(feat_names[feat[feat_show_idx[0]]])
                axs[axi].set_ylabel(feat_names[feat[feat_show_idx[1]]])
                axi += 1

                # Predicted
                for label in np.unique(np.concatenate(labels)):
                    subdata = np.transpose(predicted[:, feat[feat_show_idx]][labels[j] == label])
                    axs[axi].scatter(*subdata, label=label, s=5.)
                axs[axi].set_title(f'Imputed {dataset_names[j]}')
                axs[axi].set_xlabel(feat_names[feat[feat_show_idx[0]]])
                axs[axi].set_ylabel(feat_names[feat[feat_show_idx[1]]])
                axi += 1

                # NN Predicted
                if not skip_nn:
                    nn_predicted = predict_nn(
                        torch.tensor(dataset[i]).float(),
                        torch.tensor(dataset[j]).float())
                    for label in np.unique(np.concatenate(labels)):
                        subdata = np.transpose(
                            nn_predicted[:, feat[feat_show_idx]][labels[j] == label])
                        axs[axi].scatter(*subdata, label=label, s=5.)
                    axs[axi].set_title('NN Predicted')
                    axs[axi].set_xlabel(feat_names[feat[feat_show_idx[0]]])
                    axs[axi].set_ylabel(feat_names[feat[feat_show_idx[1]]])
                    axi += 1

                # Correlation per feature
                xaxis = np.linspace(0, 1, len(corr_per_feature))
                axs[axi].plot(xaxis, corr_per_feature[feat[::-1]])
                axs[axi].fill_between(xaxis, 0, corr_per_feature[feat[::-1]], alpha=.25)
                axs[axi].set_title('Correlation by Feature')
                axs[axi].set_xlabel('Percentile of Features')
                axs[axi].set_ylabel('Correlation')
                axi += 1

                # Correlation per sample
                xaxis = np.linspace(0, 1, len(corr_per_sample))
                axs[axi].plot(xaxis, corr_per_sample[samp[::-1]])
                axs[axi].fill_between(xaxis, 0, corr_per_sample[samp[::-1]], alpha=.25)
                axs[axi].set_title('Correlation by Sample')
                axs[axi].set_xlabel('Percentile of Samples')
                axs[axi].set_ylabel('Correlation')
                axi += 1

                # Simple model
                if not skip_simple:
                    simple_cm = ComManDo(
                        model_class=SimpleJAMIEModel, output_dim=simple_num_features)
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

                # Calibration plots
                csubfig = csubfigs[fig_idx]
                axs = csubfig.subplots(1, self.num_best_reconstructed_features)
                fig_idx += 1
                for feature_idx, ax in zip(feat[:self.num_best_reconstructed_features], axs):
                    for label in np.unique(np.concatenate(labels)):
                        true = np.transpose(actual[:, feature_idx][labels[j] == label])
                        pred = np.transpose(predicted[:, feature_idx][labels[j] == label])
                        ax.scatter(true, pred, label=label, s=5.)
                    # Plot y=x
                    lims = [
                        max(ax.get_xlim()[0], ax.get_ylim()[0]),
                        min(ax.get_xlim()[1], ax.get_ylim()[1])]
                    ax.plot(lims, lims, 'k-', alpha=0.75)
                    # Add corr info
                    ax.annotate(
                        f'r={corr_per_feature[feature_idx]:.3f}; '
                        f'p<{p_per_feature[feature_idx]:.1e}',
                        xy=(.05, .9), xycoords='axes fraction',
                        fontsize=15)

                    ax.set_title(f'{feat_names[feature_idx]}')
                    ax.set_xlabel('True Value')
                    ax.set_ylabel('Predicted Value')
        # cfig.suptitle('Modality Prediction')

    def _get_auroc_shape(self):
        scale = 1
        rows = 1
        cols = self.num_modalities
        return scale, rows, cols

    def _plot_auroc(self, cfig):
        cm_trained = self.cm_trained
        dataset = self.dataset
        dataset_names = self.dataset_names
        num_modalities = self.num_modalities

        axs = ensure_list(cfig.subplots(1, num_modalities))
        for i in range(num_modalities):
            ax = axs[i]
            pred = cm_trained.modal_predict(dataset[i], i)
            true = dataset[(i + 1) % 2]; true = 1 * (true > np.median(true))

            # fpr, tpr, _ = roc_curve(true.flatten(), pred.flatten())
            # ax.plot(fpr, tpr)
            # ax.set_xlabel('FPR')
            # ax.set_ylabel('TPR')
            # ax.set_title(f'ROC {dataset_names[(i + 1) % 2]}')

            feat_auc = []
            for pr, tr in zip(np.transpose(pred), np.transpose(true)):
                if len(np.unique(tr)) == 2:
                    feat_auc.append(roc_auc_score(tr, pr))

            sorted = np.sort(feat_auc)
            xaxis = np.linspace(0, 1, len(sorted))
            ax.plot(xaxis, sorted)
            ax.fill_between(xaxis, 0, sorted, alpha=.25)
            ax.set_title(f'AUROC by {dataset_names[(i + 1) % 2]}')
            ax.set_xlabel('Feature Percentile')
            ax.set_ylabel('AUROC')



def test_closer(
    integrated_data,
    distance_metric=lambda x: pairwise_distances(x, metric='euclidean'),
):
    """Test fraction of samples closer than the true match"""
    # ASDF: 3+ datasets and non-aligned data
    assert len(integrated_data) == 2, 'Two datasets are supported for FOSCTTM'

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


def test_LabelTA(integrated_data, datatype, k=5, return_k=False):
    """Modified version of UnionCom ``test_LabelTA`` to return acc"""
    if k is None:
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


def plot_regular(*args, **kwargs):
    plot_integrated(*args, **kwargs, separate_dim=True)


def plot_integrated(
    data,
    labels,
    names=None,
    legend=False,
    remove_outliers=False,
    n_components=2,
    separate_dim=False,
    square=False,
    method='umap',
    n_neighbors=None,
):
    assert method in ('pca', 'umap')
    method_names = {'pca': 'PC', 'umap': 'UMAP'}
    assert n_components in (2, 3), 'Only supports 2d and 3d at this time.'
    proj_method = '3d' if n_components == 3 else None
    axs = []
    for i, (dat, lab) in enumerate(zip(data, labels)):
        ax = plt.gcf().add_subplot(1, len(data), i+1, projection=proj_method)
        axs.append(ax)
        if i == 0 or separate_dim:
            if method == 'pca':
                red = PCA(n_components=n_components)
                if separate_dim:
                    red.fit(dat)
                else:
                    red.fit(np.concatenate(data, axis=0))
            elif method == 'umap':
                red = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=min(200, dat.shape[0] - 1) if n_neighbors is None else n_neighbors,
                    min_dist=.5)
                if separate_dim:
                    red.fit(dat)
                else:
                    red.fit(np.concatenate(data, axis=0))
        plot_data = red.transform(dat)
        if remove_outliers:
            filter = outliers(plot_data)
        for l in np.unique(np.concatenate(labels)):
            data_subset = np.transpose(plot_data[lab == l])
            if remove_outliers:
                data_subset[~filter[lab == l].T] = np.nan
            ax.scatter(*data_subset, s=3e3*(1/dat.shape[0]), label=l)
        if i == 1 and legend:
            ax.legend()
        if names is not None:
            ax.set_title(names[i])
        ax.set_xlabel(f'{method_names[method]}-1')
        ax.set_ylabel(f'{method_names[method]}-2')
        if n_components == 2 and square:
            ax.set_aspect('equal')
        elif n_components == 3:
            ax.set_zlabel(f'{method_names[method]}-3')
            if square:
                # https://stackoverflow.com/a/13701747
                X, Y, Z = np.transpose(plot_data)
                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
                for xb, yb, zb in zip(Xb, Yb, Zb):
                   ax.plot([xb], [yb], [zb], 'w')
    if not separate_dim:
        axs_xlim = np.array([ax.get_xlim() for ax in axs])
        axs_ylim = np.array([ax.get_ylim() for ax in axs])
        new_xlim = (axs_xlim.min(axis=0)[0], axs_xlim.max(axis=0)[1])
        new_ylim = (axs_ylim.min(axis=0)[0], axs_ylim.max(axis=0)[1])
        for ax in axs:
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)


def plot_accuracy(data, labels, names, colors=None):
    types = [np.unique(type, return_inverse=True)[1] for type in labels]
    # Metric by Algorithm
    acc_dict = {
        'Algorithm': names,
        'LTA': [],
        'FOSCTTM': [],
    }
    for i in range(len(data)):
        with contextlib.redirect_stdout(None):
            lta, k = test_LabelTA(data[i], types, return_k=True)
            acc_dict['LTA'].append(lta)
            acc_dict['FOSCTTM'].append(test_closer(data[i]))
    acc_dict[f'LTA (k={k})'] = acc_dict.pop('LTA')
    df = pd.DataFrame(acc_dict).melt(
        id_vars='Algorithm',
        value_vars=list(set(acc_dict.keys()) - set('Algorithm')))

    for i, v in enumerate(np.unique(df['variable'])):
        ax = plt.gcf().add_subplot(2, 1, i+1)
        pl = sns.barplot(
            data=df[df['variable'] == v],
            x='Algorithm',
            y='value',
            ax=ax,
            palette=colors,
        )
        ax.set_ylabel(v)
        ax.set_xlabel(None)


def plot_accuracy_table(data, labels, names, exclude=[]):
    types = [np.unique(type, return_inverse=True)[1] for type in labels]
    # Metric by Algorithm
    acc_dict = {
        'Algorithm': [names[i] for i in range(len(data)) if i not in exclude],
        'LTA': [],
        'FOSCTTM': [],
    }
    for i in range(len(data)):
        if i in exclude:
            continue
        with contextlib.redirect_stdout(None):
            lta, k = test_LabelTA(data[i], types, return_k=True)
            acc_dict['LTA'].append(lta)
            acc_dict['FOSCTTM'].append(test_closer(data[i]))
    acc_dict[f'LTA (k={k})'] = acc_dict.pop('LTA')
    df = pd.DataFrame(acc_dict)
    df.index = df['Algorithm']
    df = df[set(df.columns) - {'Algorithm'}]
    df = df.transpose()
    raw_values = df.to_numpy().copy()
    df = df.transpose()
    df['FOSCTTM'] *= -1
    df = df.transpose()
    df = df.sub(df.min(axis=1), axis=0)
    df = df.div(df.max(axis=1), axis=0)

    ax = plt.gcf().add_subplot(1, 1, 1)

    # Heatmap
    # df = df * .4
    # df = df + .3
    # pl = sns.heatmap(df, annot=raw_values, cbar=False, vmin=0, vmax=1,
    #                  cmap=sns.diverging_palette(10, 133, as_cmap=True))
    # ax.set_xlabel(None)

    # Corrplot
    # https://stackoverflow.com/questions/59381273/heatmap-with-circles-indicating-size-of-population
    df = df * .6 + .4
    df = df / 2
    x, y = np.meshgrid(np.arange(df.shape[1]), np.arange(df.shape[0]))
    circles = [
        plt.Circle((i, j), radius=r)
        for i, j, r in zip(x.flat, y.flat, df.to_numpy().flatten())
    ]
    col = PatchCollection(circles, facecolor='lightsteelblue')
    for i, j, r in zip(x.flat, y.flat, raw_values.flatten()):
        plt.text(i, j, f'{r:.2f}', color='black', ha='center', va='center')
    ax.add_collection(col)
    ax.set(
        xticks=np.arange(df.shape[1]),
        yticks=np.arange(df.shape[0]),
        xticklabels=df.columns,
        yticklabels=df.index,
    )
    ax.set_xticks(np.arange(df.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(df.shape[0]+1)-0.5, minor=True)
    ax.axis('square')
    ax.set_ylim(-.5, df.shape[0]-.5)
    ax.set_xlim(-.5, df.shape[1]-.5)
    ax.grid(which='minor')


def plot_accuracy_graph(data, labels, names, colors=None, shapes=None):
    if colors is None:
        colors = len(data) * [None]
    if shapes is None:
        shapes = len(data) * [None]

    types = [np.unique(type, return_inverse=True)[1] for type in labels]
    # Metric by Algorithm
    acc_dict = {
        'Algorithm': names,
        'FOSCTTM': [],
        'LTA': [],
    }
    for i in range(len(data)):
        with contextlib.redirect_stdout(None):
            acc_dict['FOSCTTM'].append(test_closer(data[i]))
            lta, k = test_LabelTA(data[i], types, return_k=True)
            acc_dict['LTA'].append(lta)
    df = pd.DataFrame(acc_dict)
    df.index = df['Algorithm']
    df = df[list(set(df.columns) - {'Algorithm'})]
    df = df.transpose()

    # Reporting
    print(df)

    # Calculate discontinuities
    bounds = []
    for vals in [df.transpose()['FOSCTTM'], df.transpose()['LTA']]:
        bounds.append([])
        max_dist = .2
        pad = .095

        sorted_vals = np.sort(vals)
        min_val = sorted_vals[0]
        max_val = sorted_vals[0]
        for val in sorted_vals[1:]:
            if val - max_val > max_dist:
                bounds[-1].append((min_val - pad, max_val + pad))
                min_val = max_val = val
            else:
                max_val = val
        bounds[-1].append((min_val - pad, max_val + pad))

    # Reverse x axis
    bounds[0] = [e[::-1] for e in bounds[0]][::-1]

    # Plot
    bax = brokenaxes(
        xlims=bounds[0],
        ylims=bounds[1],
        hspace=.15,
        wspace=.15,
    )
    for col, c, m in zip(df.columns, colors, shapes):
        bax.scatter(df[col]['FOSCTTM'], df[col]['LTA'], c=c, marker=m, s=200.)
    bax.set_xlabel('FOSCTTM', labelpad=30)
    bax.set_ylabel(f'LTA (k={k})', labelpad=45)

    # Add text
    tbs = []
    for i, row in df.transpose().iterrows():
        idx = (len(bounds[1]) - 1) * len(bounds[0])
        for bound in bounds[0]:
            if min(bound) <= row['FOSCTTM'] <= max(bound):
                break
            idx += 1
        else:  # Executes if no break
            assert False, 'Value not within `bound`'
        for bound in bounds[1]:
            if min(bound) <= row['LTA'] <= max(bound):
                break
            idx -= len(bounds[0])
        else:  # Executes if no break
            assert False, 'Value not within `bound`'
        ax = bax.axs[idx]

        coord = [row['FOSCTTM'], row['LTA']]
        convert = ax.transData + bax.big_ax.transData.inverted()
        coord = convert.transform(coord)
        tbs.append(
            bax.big_ax.text(
                *coord,
                i.replace('\n', ' '),
                ha='center',
                va='center',
                # transform=ax.transAxes,
        ))
    adjust_text(
        tbs,
        force_points=2.,
        arrowprops=dict(
            arrowstyle='-',
            color='black',
            shrinkA=0,
            shrinkB=10,
        ),
    )


def plot_silhouette(data, labels, names, modal_names, colors=None):
    types = [np.unique(type, return_inverse=True)[1] for type in labels]

    axs = plt.gcf().subplots(1, len(data[0]))
    if len(data[0]) == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        # Calculate coefficients
        df = pd.DataFrame(columns=['Algorithm', 'Type', 'Silhouette Coefficient'])
        for j in range(len(data)):
            coefs = silhouette_samples(data[j][i], types[i])
            for l in np.unique(np.concatenate(labels)):
                for value in coefs[labels[i] == l]:
                    df = df.append({
                        'Algorithm': names[j],
                        'Type': l,
                        'Silhouette Coefficient': value,
                    }, ignore_index=True)

        # Plot
        sns.boxplot(
            data=df,
            x='Type',
            y='Silhouette Coefficient',
            hue='Algorithm',
            ax=ax,
            palette=colors,
        )
        for j in range(len(np.unique(np.concatenate(labels)))-1):
            ax.axvline(x=j+.5, color='black', linestyle='--')
        ax.set_title(f'Silhouette Coefficients ({modal_names[i]})')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.get_legend().remove()


def _plot_auroc(imputed_data, data, modal_names, ax, i=0, names=None, max_features=100_000):
    total_features = min(data[i].shape[1], max_features)
    # Samples by default
    feat_idx = np.random.choice(data[i].shape[1], total_features, replace=False)

    feat_auc = []
    for im in imputed_data:
        pred = im[i]
        true = data[i]; true = 1 * (true > np.median(true))

        temp = []
        for pr, tr in zip(np.transpose(pred)[feat_idx], np.transpose(true)[feat_idx]):
            if len(np.unique(tr)) == 2:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    temp.append(roc_auc_score(tr, pr))
        feat_auc.append(temp)
    _plot_auroc_correlation_template(ax, feat_auc, names, 'AUROC', modal_names[i])

def _plot_correlation(imputed_data, data, modal_names, ax, i=0, names=None, max_features=100_000):
    total_features = min(data[i].shape[1], max_features)
    # Samples by default
    feat_idx = np.random.choice(data[i].shape[1], total_features, replace=False)

    feat_corr = []
    for im in imputed_data:
        pred = im[i]
        true = data[i]

        temp = []
        for pr, tr in zip(np.transpose(pred)[feat_idx], np.transpose(true)[feat_idx]):
            if len(np.unique(tr)) > 1:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    temp.append(r_regression(np.reshape(pr, (-1, 1)), tr)[0])
                # p_per_feature.append(f_regression(predicted[:, [k]], actual[:, k])[1][0])
        feat_corr.append(temp)
    _plot_auroc_correlation_template(ax, feat_corr, names, 'Correlation', modal_names[i])


def _plot_auroc_correlation_template(ax, feat, names, suptitle, modal_name, plot_type='scatter'):
    assert plot_type in ('scatter', 'density')

    if plot_type == 'scatter':
        # Scatterplot
        if len(feat[0]) > 100:
            s = 3
        else:
            s = 10
        ax.scatter(*feat, facecolor='black', edgecolor='none', s=s)
        ax.axis('square')
        lcolor='red'
    elif plot_type == 'density':
        # Density plot
        # https://www.python-graph-gallery.com/85-density-plot-with-matplotlib
        from scipy.stats import kde
        nbins = 300
        x, y = [np.array(f) for f in feat]
        proc = np.stack([x, y], axis=0)
        proc = proc[:, ~np.isnan(proc).any(axis=0)]
        proc = proc[:, ~np.isinf(proc).any(axis=0)]
        x, y = proc[0], proc[1]
        k = kde.gaussian_kde([x,y])
        MIN = min(x.min(), y.min())
        MAX = min(x.max(), y.max())
        xi, yi = np.mgrid[MIN:MAX:nbins*1j, MIN:MAX:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Greys')
        lcolor='red'

    ax.set_title(f'{suptitle} - {modal_name}')
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])

    # Plot y=x
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, '-', color=lcolor, alpha=0.75)

    # Text output
    # would use transform=ax.transAxes, but warps
    gre = sum(np.greater(feat[1], feat[0]))
    ax.text(.05, .9, gre, ha='left', va='center', transform=ax.transAxes, backgroundcolor='white')
    les = sum(np.greater(feat[0], feat[1]))
    ax.text(.95, .2, les, ha='right', va='center', transform=ax.transAxes, backgroundcolor='white')
    n = gre + les  # len(feat[0]) doesn't account for NAs
    # NULL hypothesis 50/50, one-tailed
    p_value = sum(2**(math.log(math.comb(n, i), 2) - n) for i in range(n+1) if i >= gre)
    if p_value > .5:
        p_value = 1 - p_value
    p_value *= 2
    # p_value = stats.ranksums(*feat, 'less')[1]
    ax.text(.95, .1, f'p-value: {p_value:.2E}', ha='right', va='center', transform=ax.transAxes, backgroundcolor='white')


def plot_auroc(*args, **kwargs):
    axs = plt.gcf().subplots(1, 2)
    for i, ax in enumerate(axs):
        _plot_auroc(*args, ax=ax, i=i, **kwargs)


def plot_correlation(*args, **kwargs):
    axs = plt.gcf().subplots(1, 2)
    for i, ax in enumerate(axs):
        _plot_correlation(*args, ax=ax, i=i, **kwargs)


def plot_auroc_correlation(*args, index=0, **kwargs):
    axs = plt.gcf().subplots(1, 2)
    _plot_auroc(*args, ax=axs[0], i=index, **kwargs)
    _plot_correlation(*args, ax=axs[1], i=index, **kwargs)


def plot_distribution_alone(
    datasets,
    labels,
    feature_limit=2,
    title=None,
    fnames=None,
    gcf=None,
    rows=2,
    remove_outliers=True,
    equal_axes=False,
    feature_dict={},
    **kwargs,
):
    datasets = [np.array(d) for d in datasets]
    fnames = [
        fnames[i]
        if fnames[i] is not None else [f'Feature {i}' for i in range(len(df.columns))]
        for i in range(2)]
    if gcf is None:
        gcf = plt.gcf()

    # Destructively limit data
    names = ['Measured', 'Imputed']
    feature_limit = feature_limit if feature_limit is not None else datasets[0].shape[1]
    feature_idx = sort_by_interest(datasets,
                                   limit=feature_limit,
                                   remove_outliers=remove_outliers)[1]
    datasets = [data[:, feature_idx] for data in datasets]
    # if remove_outliers:
    #     filter = outliers(datasets[0], verbose=True)
    #     for i in range(len(datasets)):
    #         datasets[i][filter] = np.nan
    #         print(np.max(datasets[i][~np.isnan(datasets[i])]))
    for i in range(len(fnames)):
        fnames[i] = fnames[i][feature_idx]
        for j in range(len(fnames[i])):
            if fnames[i][j] in feature_dict:
                fnames[i][j] = feature_dict[fnames[i][j]]

    # Reporting
    for i in range(datasets[0].shape[1]):
        print(f'{fnames[0][i]}: {jensen_shannon_from_array([d[:, i] for d in datasets])}')

    # Distribution preview
    axs = []
    for i in range(2):
        if i == 0:
            ax = gcf.add_subplot(rows, 1, rows-1+i)
        elif i == 1:
            ax = gcf.add_subplot(rows, 1, rows-1+i, sharex=ax)
        else:
            assert False, 'Unexpected number of subplots'
        axs.append(ax)
        df = pd.DataFrame(datasets[i])
        fname = fnames[i]
        fname = np.array(fname)
        df.columns = fname
        df.columns.name = None
        df['_type'] = labels[i]
        df['_sample'] = df.index

        id_vars = ['_sample', '_type']
        df = df.melt(
            id_vars=id_vars,
            value_vars=list(set(df.keys()) - set(id_vars)))
        df = df.rename(columns={'variable': 'Variable', 'value': 'Value', '_type': 'Type'})
        df['fsorted'] = [np.argwhere(fname==x)[0][0] for x in df['Variable']]
        df = df.sort_values('fsorted')

        # Plot
        sns.boxplot(
            data=df,
            x='Variable',
            y='Value',
            hue='Type',
            ax=ax,
            # showfliers=not remove_outliers,
        )
        for j in range(feature_limit-1):
            ax.axvline(x=j+.5, color='black', linestyle='--')
        ax.set_xlabel(None)
        if i == 0:
            ax.set_xticks([])
            ax.set_title(f'Sample Feature Distributions ({title})')
            # ax.axhline(y=ax.get_ylim()[0], color='black', linestyle='-', linewidth=5)
        else:
            ax.set_title(None)
        ax.set_ylabel(names[i])
        ax.legend([], [], frameon=False)
    if remove_outliers:
        for i in range(len(axs)):
            ax = axs[i] if not equal_axes else axs[0]
            d = datasets[i] if not equal_axes else datasets[0]
            new_ylim = outliers(d, return_limits=True)[1]
            stretch=1.5
            new_ylim = (
                np.min(new_ylim[0]-stretch*new_ylim[2]),
                np.max(new_ylim[1]+stretch*new_ylim[2]))
            new_ylim = (max(new_ylim[0], ax.get_ylim()[0]), min(new_ylim[1], ax.get_ylim()[1]))
            axs[i].set_ylim(new_ylim)
    for ax in axs:
        set_yticks(ax, 4)
    plt.gcf().subplots_adjust(hspace=0)


def plot_distribution(datasets, labels, feature_limit=3, title=None, **kwargs):
    datasets = [np.array(d) for d in datasets]

    # Grid
    ax = plt.gcf().add_subplot(3, 1, 1)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])
    ax.set_subplotspec(gs[0])
    # Distribution similarity
    plot_distribution_similarity(
        datasets,
        labels,
        suptitle=title,
        ax=ax,
        square=False,
        legend=False,
        **kwargs)
    set_yticks(ax, 2)
    ax.set_xticks([])
    ax.set_xlim([0, 1])
    ax.set_ylabel('Simulated')

    plot_distribution_alone(datasets, labels, rows=3, title=None, **kwargs)
    plt.gcf().subplots_adjust(hspace=0)


def plot_distribution_similarity(
    datasets,
    labels,
    suptitle=None,
    title=None,
    max_features=100,
    relative=True,
    label_cells=True,
    legend=False,
    square=True,
    ax=None,
    **kwargs,
):
    assert datasets[0].shape[1] == datasets[1].shape[1]
    datasets = [np.array(d) for d in datasets]

    # Assumes aligned datasets
    total_features = min(datasets[0].shape[1], max_features)
    # Samples by default
    feat_idx = np.random.choice(datasets[0].shape[1], total_features, replace=False)
    if ax is None:
        ax = plt.gcf().add_subplot(1, 1, 1)
    distances = {}
    for l in np.unique(labels):
        distances[l] = []
        for f in feat_idx:
            dist = jensen_shannon_from_array([datasets[i][labels[i] == l, f] for i in range(len(datasets))])
            # Cast nan to 0 for visualization
            if np.isnan(dist):
                dist = 1
            distances[l].append(1 - dist)

    # Sort by performance
    total = 0
    for l, v in distances.items():
        total += np.array(v)
    total /= len(distances.keys())
    sort_idx = np.argsort(total)[::-1]

    # Reporting
    all_values = np.concatenate(list(distances.values()))
    print(f'Mean: {1 - np.mean(all_values)}')
    print(f'Std: {np.std(np.concatenate(list(distances.values())))}')

    # Plot
    for l, v in distances.items():
        # ax.plot(range(total_features), np.array(v)[sort_idx], label=l)
        # pct = np.array([sum(np.array(v) <= i) / len(v) for i in np.array(v)])
        pct = np.linspace(0, 1, len(v))
        sort_idx = np.argsort(v)
        ax.plot(pct, np.array(v)[sort_idx], label=l if label_cells else '_nolegend_')
    # ax.plot(range(total_features), total[sort_idx], label='Cumulative', linewidth=6, color='black')
    # pct = np.array([sum(total <= i) / len(total) for i in total])
    pct = np.linspace(0, 1, len(total))
    sort_idx = np.argsort(total)
    ax.plot(pct, total[sort_idx], label='Cumulative', linewidth=6, color='black')

    ax.set_xlabel('Percentile')
    ax.set_ylabel(f'{title} Similarity')
    # plt.tick_params(
    #     axis='x',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     labelbottom=False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(suptitle)
    if square:
        ax.set_aspect('equal', adjustable='box')
    if legend:
        ax.legend()
    else:
        ax.legend([], [], frameon=False)


def plot_impact(
    values,
    fnames,
    baseline,
    ylabel='LTA',
    max_features=None,
    background_pct=.3,
    sort='mixed-min',
    color=None,
    max_name_len=10,
):
    num_features = len(values) if max_features is None else max_features
    num_features = min(len(values), num_features)
    if sort is not None:
        if sort == 'min':
            sort = np.argsort(values)
        elif sort == 'max':
            sort = np.argsort(values)[::-1]
        elif sort.split('-')[0] == 'mixed':
            if sort.split('-')[1] == 'max':
                var1 = np.argsort(values)[::-1]
            elif sort.split('-')[1] == 'min':
                var1 = np.argsort(values)
            else:
                assert False, f'Invalid sort method \'{sort}\' provided.'
            var1 = var1[:int((1 - background_pct) * num_features)]
            var2 = np.random.choice(
                list(set(list(range(len(values)))) - set(var1)),
                num_features - len(var1),
                replace=False)
            sort = np.concatenate([var1, var2])
            np.random.shuffle(sort)
        else:
            assert False, f'Invalid sort method \'{sort}\' provided.'
        values = values[sort]
        fnames = fnames[sort]
    values = values[:num_features]
    fnames = fnames[:num_features]
    fnames = [f[:max_name_len] for f in fnames]

    ax = plt.gcf().add_subplot(1, 1, 1)
    sns.barplot(x=fnames, y=values, ax=ax, color=color)
    ax.axhline(y=baseline, color='red', linewidth=3)
    ax.set_ylabel(ylabel)
    yrange = max(values) - min(values)
    ymin, ymax = max(min(values) - 1. * yrange, 0), min(max(values) + 1. * yrange, 1)
    ax.set_ylim([ymin, ymax])
    plt.xticks(rotation=80)


def evaluate_impact(function, perf_function, in_data, true, features=None, idx=None, mode='replace', scan=None, scan_samples=500):
    assert mode in ['replace', 'keep']

    testing_idx = idx if idx is not None else np.array(range(in_data.shape[1]))

    in_data = in_data.copy()
    background = in_data.mean(0)

    # Calculate baseline
    logits = function(in_data)
    baseline = perf_function(logits, true)

    if scan is not None:
        print('Performing preliminary scan...')
        sample_idx = np.random.choice(in_data.shape[0], scan_samples, replace=False)
        true_mini = true[sample_idx] if true is not None else None
        performance = _evaluate_impact_helper(function, perf_function, in_data[sample_idx, :], true_mini, background, baseline, testing_idx, mode, features=features)
        if mode == 'keep':
            performance = -performance
        testing_idx = testing_idx[np.argsort(testing_idx)[:scan]]
    print('Finding important features...')
    performance = _evaluate_impact_helper(function, perf_function, in_data, true, background, baseline, testing_idx, mode, features=features)
    print('Done!')

    return baseline, performance, testing_idx


def _evaluate_impact_helper(function, perf_function, in_data, true, background, baseline, testing_idx, mode, features=None, check_best=10):
    performance = []
    best_idx = -1
    best_perf = -np.inf
    best_str = ''
    for i, idx in enumerate(testing_idx):
        # CLI
        if (i+1) % check_best == 0 and len(performance) > 0:
            if mode == 'replace':
                best_idx = np.argmax(-np.array(performance))
            elif mode == 'keep':
                best_idx = np.argmax(performance)
            best_perf = performance[best_idx]
            best_str = features[testing_idx[best_idx]] if features is not None else 'NA'
        prog_str = math.floor(50*(i+1)/len(testing_idx)) * '|'
        print(
            f'{i+1:>{len(str(len(testing_idx)))}}/{len(testing_idx)} [{prog_str:<50}] - '
            f'Current Best: {best_perf:.5f}, {best_str}'
            , end='\r')

        mod_data = in_data
        # Replace one
        if mode == 'replace':
            replace_idx = idx
        elif mode == 'keep':
            replace_idx = [i!=idx for i in range(mod_data.shape[1])]
        old_data = mod_data[:, replace_idx]
        mod_data[:, replace_idx] = background[replace_idx]

        # Predict
        logits = function(mod_data)
        # logits = current_cm.modal_predict(mod_data, mod0)[:, target]  # Imputation
        # logits = current_cm.test_LabelTA(2*[current_cm.transform_one(mod_data, mod0)], labels)  # LTA

        # Repair
        mod_data[:, replace_idx] = old_data

        # Record
        # perf = stats.pearsonr(logits, out_data)[0]  # Imputation
        # perf = logits  # LTA
        perf = perf_function(logits, true)
        if np.isnan(perf):
            perf = -np.inf
        performance.append(perf)
    print()
    return np.array(performance)
