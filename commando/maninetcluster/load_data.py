##########################################################################
### Load Fly & Worm Data and Prepare X, Y, Corr for Manifold Alignment ###
##########################################################################
import pandas as pd
from os import sys

datOrtho = pd.read_csv(sys.argv[3]) # Modencode.merged.orth20120611_wfh_1to1_comm.csv

datFly = pd.read_csv(sys.argv[1]).iloc[:, 0:13] # fly_data_all_30stages.csv
print(datFly.shape)
low_expr_mask = datFly.iloc[:, 1:13].sum(axis = 1) < 1
datFly = datFly[~low_expr_mask]
print(datFly.shape)

iy = datOrtho.set_index("dmel").index
iX = datFly.set_index("Gene").index
datFly = datFly[iX.isin(iy)]
print(datFly.shape)

datWorm = pd.read_csv(sys.argv[2]) # worm_data_all_25stages_foreigensys.csv
print(datWorm.shape)
low_expr_mask = datWorm.iloc[:, 1:26].sum(axis = 1) < 1
datWorm = datWorm[~low_expr_mask]
print(datWorm.shape)

iy = datOrtho.set_index("celegans").index
iX = datWorm.set_index("Unnamed: 0").index
datWorm = datWorm[iX.isin(iy)]
print(datWorm.shape)

corr = pd.crosstab(datOrtho.celegans, datOrtho.dmel)

iy = datWorm.set_index("Unnamed: 0").index
iX = corr.index
corr = corr[iX.isin(iy)]

iy = datFly.set_index("Gene").index
iX = corr.columns
corr = corr.iloc[:, iX.isin(iy)]

cols = datFly.set_index("Gene").index.tolist()
corr = corr[cols]

idx = datWorm.set_index("Unnamed: 0").index
corr = corr.reindex(idx)


datFly = datFly.set_index("Gene")
datFly = datFly.reindex(datOrtho["dmel"].as_matrix()).dropna()
datWorm = datWorm.set_index("Unnamed: 0")
datWorm = datWorm.reindex(datOrtho["celegans"].as_matrix()).dropna()

corr = pd.crosstab(datOrtho.celegans, datOrtho.dmel)

iy = datWorm.index
iX = corr.index
corr = corr[iX.isin(iy)]

iy = datFly.index
iX = corr.columns
corr = corr.iloc[:, iX.isin(iy)]

cols = datFly.index.tolist()
corr = corr[cols]

idx = datWorm.index
corr = corr.reindex(idx)

store = pd.HDFStore('store.h5')
store['worm'] = datWorm
store['fly'] = datFly
store.close()