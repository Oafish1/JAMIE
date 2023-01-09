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
    davies_bouldin_score, r2_score, roc_auc_score, roc_curve,
    silhouette_samples)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
import torch
import umap

from .jamie import JAMIE
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
        cm = JAMIE(P=np.diag(random_diag), **kwargs)
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
    """Plot differing modalities"""
    plot_integrated(*args, **kwargs, separate_dim=True)


def plot_integrated(
    data,
    labels,
    names=None,
    legend=False,
    remove_outliers=False,
    n_components=2,
    hybrid_components=4_096,
    separate_dim=False,
    square=False,
    method='umap',
    n_neighbors=None,
    seed=42,
):
    """Plot integrated data"""
    assert method in ('pca', 'umap', 'hybrid')
    method_names = {'pca': 'PC', 'umap': 'UMAP', 'hybrid': 'PC-UMAP'}
    assert n_components in (2, 3), 'Only supports 2d and 3d at this time.'
    proj_method = '3d' if n_components == 3 else None

    if method == 'hybrid':
        # Pre-process with PCA
        red = PCA(n_components=hybrid_components)
        data = [red.fit_transform(dat) for dat in data]

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
            elif method in ('umap', 'hybrid'):
                red = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=min(200, dat.shape[0] - 1) if n_neighbors is None else n_neighbors,
                    min_dist=.5,
                    random_state=seed)
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
    """Compare accuracies of methods using barplots"""
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
    """Compare accuracies of methods using a table"""
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
    """Compare accuracies of methods using a graph"""
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
    """Plot and compare silhouette widths for different integration results"""
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
    """Plot AUROC by feature for imputation on binarized data"""
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
    """Plot correlation by feature for imputed data"""
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
    """General purpose plotting template for AUROC and correlation"""
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


def plot_sample(true, imputed, name, modal_name, suptitle=None, sample_idx=None, color='blue', scale=None, plot_type='scatter'):
    ax = plt.gca()

    # Format features
    feat = [true, imputed]

    # Calculate p-value and r^2
    if sample_idx is None:
        r2 = []
        p_value = []
        for tru, imp in zip(*feat):
            r2.append(r2_score(tru, imp))
            p_value.append(stats.pearsonr(tru, imp)[1])
        r2 = np.array(r2)
        p_value = np.array(p_value)
        sample_idx = np.argmax(r2)
        # sample_idx = np.argsort(r2)[len(np.argsort(r2))//2]
        r2 = r2[sample_idx]
        p_value = p_value[sample_idx]
    else:
        r2 = r2_score(true[sample_idx], imputed[sample_idx])
        _, p_value = stats.pearsonr(true[sample_idx], imputed[sample_idx])

    # Plot
    assert plot_type in ('scatter', 'density')

    if plot_type == 'scatter':
        # Scatterplot
        if feat[0].shape[1] > 100:
            s = 5
        else:
            s = 15
        ax.scatter(*[f[sample_idx] for f in feat], facecolor=color, edgecolor='none', s=s)
    elif plot_type == 'density':
        # Density plot
        # https://www.python-graph-gallery.com/85-density-plot-with-matplotlib
        from scipy.stats import kde
        nbins = 300
        x, y = [np.array(f[sample_idx]) for f in feat]
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

    ax.axis('square')
    ax.set_title(f'{suptitle} - {modal_name}' if suptitle is not None else f'Cell - {modal_name}')
    ax.set_xlabel('Measured')
    ax.set_ylabel(name)

    # Set limits
    maxlim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.set_xlim(maxlim)
    ax.set_ylim(maxlim)
    if scale is not None:
        ax.set_xscale(scale)
        ax.set_yscale(scale)

    # Plot y=x
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1])]
    if plot_type == 'scatter':
        ax.plot(lims, lims, '--', color='black', alpha=0.75, zorder=-1)
    elif plot_type == 'density':
        ax.plot(lims, lims, '-', color='red', alpha=0.75)

    # Text output
    ax.text(.05, .8, f'$R^2$: {r2:.2E}', ha='left', va='center', transform=ax.transAxes, backgroundcolor='white')
    ax.text(.05, .9, f'p-value: {p_value:.2E}', ha='left', va='center', transform=ax.transAxes, backgroundcolor='white')

    return sample_idx


def plot_auroc(*args, **kwargs):
    """Outward interface for plotting AUROC"""
    axs = plt.gcf().subplots(1, 2)
    for i, ax in enumerate(axs):
        _plot_auroc(*args, ax=ax, i=i, **kwargs)


def plot_correlation(*args, **kwargs):
    """Outward interface for plotting correlation"""
    axs = plt.gcf().subplots(1, 2)
    for i, ax in enumerate(axs):
        _plot_correlation(*args, ax=ax, i=i, **kwargs)


def plot_auroc_correlation(*args, index=0, **kwargs):
    """Outward interface for plotting both AUROC and correlation"""
    axs = plt.gcf().subplots(1, 2)
    _plot_auroc(*args, ax=axs[0], i=index, **kwargs)
    _plot_correlation(*args, ax=axs[1], i=index, **kwargs)


def plot_distribution_alone(
    datasets,
    labels,
    label_order=None,
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
    """Plot preview of cell-type distributions by feature"""
    datasets = [np.array(d) for d in datasets]
    fnames = [
        fnames[i]
        if fnames[i] is not None else np.array([f'Feature {i}' for i in range(datasets[i].shape[1])])
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
        df['fsorted'] = [
            np.argwhere(np.array(label_order if label_order is not None else np.unique(labels))==x)[0][0]
            for x in df['Type']]
        df = df.sort_values('fsorted')

        # Consistent box ordering
        # df.sort_values('Type')

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
            ax.set_xticklabels([])
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
    """Plot distribution preview and similarity plot together"""
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
    label_order=None,
    suptitle=None,
    title=None,
    max_features=100,
    relative=True,
    label_cells=True,
    legend=True,
    square=True,
    ax=None,
    **kwargs,
):
    """Similarity plot between cell-type distributions"""
    assert datasets[0].shape[1] == datasets[1].shape[1]
    datasets = [np.array(d) for d in datasets]

    # Assumes aligned datasets
    total_features = min(datasets[0].shape[1], max_features)
    # Samples by default
    feat_idx = np.random.choice(datasets[0].shape[1], total_features, replace=False)
    if ax is None:
        ax = plt.gcf().add_subplot(1, 1, 1)
    distances = {}
    for l in (np.unique(labels) if label_order is None else label_order):
        distances[l] = []
        for f in feat_idx:
            try:
                dist = jensen_shannon_from_array([datasets[i][labels[i] == l, f] for i in range(len(datasets))])
                # Cast nan to 0 for visualization
                if np.isnan(dist):
                    dist = 1
            except:
                dist = 0
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
    """Plot impact for various features on JAMIE model"""
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
    """Get impact values"""
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
    """Helper function for `evaluate_impact`"""
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
