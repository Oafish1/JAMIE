from commando import ComManDo
import matplotlib.pyplot as plt
import numpy as np


def test_partial(datasets, types, fraction_range=np.linspace(0, 1, 3), **kwargs):
    """Takes aligned datasets and tests accuracy based on alignment assumptions"""
    assert len(datasets[0]) == len(datasets[1]), 'Datasets must be aligned.'

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
    for k, v in acc_list.items():
        plt.plot(fraction_range, v, '.-', label=k)
    plt.xlabel("Fraction Assumed Aligned")
    plt.ylabel("Statistic")
    plt.legend()
    return acc_list
