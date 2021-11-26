import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from commando import ComManDo
from commando.utilities import visualize_mapping


data_mod1 = ad.read_h5ad('OPSCA/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_mod1.h5ad')
data_mod2 = ad.read_h5ad('OPSCA/openproblems_bmmc_multiome_phase1/openproblems_bmmc_multiome_phase1.censor_dataset.output_mod2.h5ad')
# data_mod1 = ad.read_h5ad('OPSCA/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.censor_dataset.output_mod1.h5ad')
# data_mod2 = ad.read_h5ad('OPSCA/openproblems_bmmc_cite_phase1/openproblems_bmmc_cite_phase1.censor_dataset.output_mod2.h5ad')

X1 = data_mod1.X[:, :1000]
X2 = data_mod2.X[:, :1000]

commando_out = (
    ComManDo(
        distance_mode='euclidean',
        two_step_aggregation='random',

        two_step_num_partitions=None,
        epoch_pd=1000,
        two_step_pd_large=1000,

        log_pd=1,
        two_step_log_pd=1,
    )
    # .fit_transform([data_mod1, data_mod2])
    .fit_transform([X1, X2])
)

plt.subplots(figsize=(12, 6))
plt.subplot(1, 2, 1)
visualize_mapping(commando_out, 0)
plt.subplot(1, 2, 2)
visualize_mapping(commando_out, 1)
