import contextlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import r_regression
from sklearn.metrics import silhouette_samples, silhouette_score
import torch

from .commando import ComManDo
from .utilities import ensure_list, predict_nn


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


def generate_figure(
    # Data
    cm_trained,
    cm_data,
    dataset,
    labels,
    alg_results=[],
    alg_names=[],
    dataset_names=None,
    # Style
    scale=20,
    size_bound=[.25, .5],
    vertical_scale=.75,
    # Visualizations
    reconstruction_features={},
    # Remove Visualizations
    exclude_predict=[],
    skip_partial=False,
    skip_nn=False,
):
    """Compares ComManDo with ``alg_results``"""
    # asddf: Legends may be wrong unless sorted
    assert len(alg_results) == len(alg_names), '``alg_*`` params must correspond.'

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

    # Setup
    integrated_data = [cm_data, *alg_results]
    integrated_alg_names = ['ComManDo', *alg_names]
    types = [np.unique(type, return_inverse=True)[1] for type in labels]
    num_modalities = len(dataset)
    num_algorithms = len(integrated_data)
    dataset_names = [
        dataset_names[i] if dataset_names is not None else f'Dataset {i}'
        for i in range(num_modalities)
    ]

    # Number of rows
    height_numerators = [
        1,
        1,
        1,
        1,
        (num_modalities**2 - num_modalities) - len(exclude_predict)
    ]
    # Number of columns
    height_denominators = [
        (num_modalities),
        (num_modalities * num_algorithms),
        (2 - skip_partial),
        (num_modalities),
        (5 if not skip_nn else 4),
    ]
    # Control for min/max single-row height
    height_denominators = [
        min(1/size_bound[0], max(1/size_bound[1], x)) for x in height_denominators]
    height_ratios = [n/d for n, d in zip(height_numerators, height_denominators)]
    figsize = (scale, scale * vertical_scale * sum(height_ratios))
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    subfigs = fig.subfigures(5, 1, height_ratios=height_ratios, wspace=.07)

    # Raw Data
    cfig = subfigs[0]
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
    fig.suptitle('Raw Data')

    # Integrated Data
    cfig = subfigs[1]
    csubfigs = cfig.subfigures(1, num_algorithms, wspace=.07)
    for csubfig, j in zip(csubfigs, range(num_algorithms)):
        for i in range(num_modalities):
            ax = csubfig.add_subplot(1, num_modalities, i+1)  # , projection='3d'
            for label in np.unique(np.concatenate(labels)):
                data_subset = np.transpose(integrated_data[j][i][labels[i] == label])[:2, :]
                ax.scatter(*data_subset, s=5., label=label)
            title = dataset_names[i]
            ax.set_title(title)
            ax.set_xlabel('Latent Feature 1')
            ax.set_ylabel('Latent Feature 2')
            # ax.set_zlabel('Latent Feature 3')
        csubfig.suptitle(integrated_alg_names[j])
    cfig.suptitle('Integrated Embeddings')

    # Accuracy by Partial
    cfig = subfigs[2]
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
    #
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
    cfig.suptitle('Miscellaneous Accuracy Statistics')
    csubfig_idx += 1

    # Distance by Cell Type
    # cfig = subfigs[3]
    # axs = cfig.subplots(1, num_algorithms)
    # for ax, i in zip(axs, range(num_algorithms)):
    #     lab, dat = cm_trained.test_label_dist(integrated_data[i], labels, verbose=False)
    #     # Sort to look nice
    #     idx = np.argsort(dat, axis=1)[0]
    #     dat = dat[idx, :][:, idx]
    #     lab = lab[idx]
    #     ax = sns.heatmap(dat, xticklabels=lab, yticklabels=lab, linewidth=0, cmap='YlGnBu', ax=ax)
    #     ax.set_title(integrated_alg_names[i])
    # cfig.suptitle('Distance of Medoid by Cell Type')

    # Silhouette Value Boxplots
    cfig = subfigs[3]
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
    cfig.suptitle('Silhouette Score by Cell Type')

    # Reconstruct Modality
    cfig = subfigs[4]
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
            axs = csubfig.subplots(1, 5 if not skip_nn else 4)
            fig_idx += 1

            csubfig.suptitle(f'{dataset_names[i]} -> {dataset_names[j]}')

            predicted = cm_trained.model.decoders[j](
                cm_trained.model.encoders[i](
                    torch.tensor(dataset[i]).float()
                )).detach().cpu().numpy()
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

    cfig.suptitle('Modality Prediction')

    plt.show()
