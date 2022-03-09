# -*- coding: utf-8 -*-
"""
Code for AllenSDK Dataset

@author: Jiawei Huang
"""
import os
import pandas as pd
from ipfx.dataset.create import create_ephys_data_set
from ipfx.data_set_features import extract_data_set_features
from ipfx.utilities import drop_failed_sweeps

import numpy as np
from ipfx.feature_extractor import (
    SpikeFeatureExtractor, SpikeTrainFeatureExtractor
)
import ipfx.stimulus_protocol_analysis as spa
from ipfx.epochs import get_stim_epoch
from ipfx.dataset.create import create_ephys_data_set
from ipfx.utilities import drop_failed_sweeps
import matplotlib.pyplot as plt
from ipfx.stimulus_protocol_analysis import RampAnalysis
import seaborn as sns

nwb_file = os.path.join(
    "../data",
    "sub-599387254_ses-601506492_icephys.nwb"
)
data_set= create_ephys_data_set(nwb_file)
drop_failed_sweeps(data_set)
short_square_table = data_set.filtered_sweep_table(
    stimuli=data_set.ontology.long_square_names
)

#all features
cell_features, sweep_features, cell_record, sweep_records, _, _ = \
    extract_data_set_features(data_set, subthresh_min_amp=-100.0)
print(cell_record)

ID = []
title = cell_record.keys()
res = pd.DataFrame(columns=title)
path_out = "../data/00020" 
files_out = os.listdir(path_out)
for file_out in files_out:
    path_in = path_out + "/" +file_out
    files_in = os.listdir(path_in)
    for file_in in files_in:      
        try:
            data_set = create_ephys_data_set(nwb_file=path_in + "\\" +file_in)
            drop_failed_sweeps(data_set)
            cell_features, sweep_features, cell_record, sweep_records, _, _ = \
                extract_data_set_features(data_set, subthresh_min_amp=-100.0)
        except:
            cell_record = dict.fromkeys(title,np.nan)
        res = res.append([cell_record], ignore_index=True)
        ID.append(file_in)
        print(path_in + "\\" +file_in)
res["ID"] = ID
res.to_csv("../data/efeatrue.csv",index=False,sep=',')
