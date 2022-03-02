from commando import ComManDo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch


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
        cm_data = cm.fit_transform(dataset=datasets)
        acc_list['lta'].append(cm.test_LabelTA(cm_data, types))
        acc_list['foscttm'].append(cm.test_closer(cm_data))
        print()

    # https://stackoverflow.com/a/42466319
    if plot:
        for k, v in acc_list.items():
            plt.plot(fraction_range, v, '.-', label=k)
        plt.xlabel("Fraction Assumed Aligned")
        plt.ylabel("Statistic")
        plt.legend()
    return acc_list, fraction_range


def generate_figure(
    cm_trained,
    cm_data,
    dataset,
    labels,
    alg_results=[],
    alg_names=[],
    dataset_names=None,
):
    """Compares ComManDo with ``alg_results``"""
    # asddf: Legends may be wrong unless sorted
    assert len(alg_results) == len(alg_names), '``alg_*`` params must correspond.'

    # Setup
    integrated_data = [cm_data, *alg_results]
    integrated_alg_names = ['ComManDo', *alg_names]
    types = [np.unique(type, return_inverse=True)[1] for type in labels]
    num_modalities = len(dataset)
    num_algorithms = len(integrated_data)

    fig = plt.figure(constrained_layout=True, figsize=(12, 16))
    subfigs = fig.subfigures(5, 1, wspace=.07)

    # Raw Data
    cfig = subfigs[0]
    axs = cfig.subplots(1, num_modalities)
    for ax, i in zip(axs, range(num_modalities)):
        pca_data = PCA(n_components=2).fit_transform(dataset[i])
        for label in np.unique(np.concatenate(labels)):
            pca_data_subset = np.transpose(pca_data[labels[i] == label])
            ax.scatter(*pca_data_subset, s=5., label=label)
        title = dataset_names[i] if dataset_names is not None else f'Dataset {i}'
        ax.set_title(title)
        ax.set_xlabel('PCA-1')
        ax.set_ylabel('PCA-2')
    ax.legend()
    cfig.suptitle('Raw Data')

    # Integrated Data
    cfig = subfigs[1]
    csubfigs = cfig.subfigures(1, num_algorithms, wspace=.07)
    for csubfig, j in zip(csubfigs, range(num_algorithms)):
        axs = csubfig.subplots(1, num_modalities)
        for ax, i in zip(axs, range(num_modalities)):
            for label in np.unique(np.concatenate(labels)):
                data_subset = np.transpose(integrated_data[j][i][labels[i] == label])[:2, :]
                ax.scatter(*data_subset, s=5., label=label)
            title = dataset_names[i] if dataset_names is not None else f'Dataset {i}'
            ax.set_title(title)
            ax.set_xlabel('Latent Feature 1')
            ax.set_ylabel('Latent Feature 2')
        csubfig.suptitle(integrated_alg_names[j])
    cfig.suptitle('Integrated Embeddings')

    # Distance by Cell Type
    cfig = subfigs[2]
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

    # Reconstruct Modality
    cfig = subfigs[3]
    csubfigs = cfig.subfigures(1, num_modalities**2 - num_modalities, wspace=.07)
    fig_idx = 0
    for i in range(num_modalities):
        for j in range(num_modalities):
            if i == j:
                continue
            csubfig = csubfigs[fig_idx]
            axs = csubfig.subplots(1, 2)
            fig_idx += 1

            if dataset_names is not None:
                csubfig.suptitle(f'{dataset_names[i]} -> {dataset_names[j]}')
            else:
                csubfig.suptitle(f'{i} -> {j}')

            predicted = cm_trained.model.decoders[j](
                cm_trained.model.encoders[i](
                    torch.tensor(dataset[i]).float()
                )).detach().cpu().numpy()
            actual = dataset[j]

            # asdf: Use PCA
            # Real
            for label in np.unique(np.concatenate(labels)):
                subdata = np.transpose(actual[:, :2][labels[j] == label])
                axs[0].scatter(*subdata, label=label, s=5.)
            axs[0].set_title(f'Actual {dataset_names[j]}')
            axs[0].set_xlabel('Latent Feature 1')
            axs[0].set_ylabel('Latent Feature 2')

            # Predicted
            for label in np.unique(np.concatenate(labels)):
                subdata = np.transpose(predicted[:, :2][labels[j] == label])
                axs[1].scatter(*subdata, label=label, s=5.)
            axs[1].set_title('ComManDo Predicted')
            axs[1].set_xlabel('Latent Feature 1')
            axs[1].set_ylabel('Latent Feature 2')
    cfig.suptitle('Modality Prediction')

    # Accuracy by Partial
    cfig = subfigs[4]
    csubfigs = cfig.subfigures(1, 2, wspace=.07)
    ax = csubfigs[0].subplots(1, 1)
    acc_list, fraction_range = test_partial(dataset, types, plot=False)
    for k, v in acc_list.items():
        ax.plot(fraction_range, v, '.-', label=k)
    ax.set_title('Accuracy by Partial Alignment')
    ax.set_xlabel('Fraction Assumed Aligned')
    ax.set_ylabel('Statistic')
    ax.legend()

    ax = csubfigs[1].subplots(1, 1)
    acc_dict = {
        'Algorithm': integrated_alg_names,
        'Label Transfer Accuracy': [],
        'FOSCTTM': [],
    }
    for i in range(num_algorithms):
        acc_dict['Label Transfer Accuracy'].append(
            cm_trained.test_LabelTA(integrated_data[i], types))
        acc_dict['FOSCTTM'].append(
            cm_trained.test_closer(integrated_data[i]))
    df = pd.DataFrame(acc_dict).melt(
        id_vars=['Algorithm'],
        value_vars=['Label Transfer Accuracy', 'FOSCTTM'])
    sns.barplot(data=df, x='variable', y='value', hue='Algorithm', ax=ax,
                palette='dark', alpha=.6)
    ax.set_title('Metric by Algorithm')
    ax.set_xlabel('Fraction Assumed Aligned')
    ax.set_ylabel(None)
    cfig.suptitle('Miscellaneous Accuracy Statistics')

    plt.show()
