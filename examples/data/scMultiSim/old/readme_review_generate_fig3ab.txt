We used scMultiSim [cite] simulator with the default settings “GRN_params_1139” to generate data of two modalities. The first modality has RNA counts of 500 cells with 1250 gene expression features. The second modality has ATAC-seq of 500 cells with 3750 features.

modality 1: n_samples * n_feature = 500 cells x 1250 gene expression features
modality 2: n_samples * n_feature = 500 cells x 3750 features

2022.12.21_adding nose
intrinsic.noise  (default: 1)
    The weight assigned to the random sample from the Beta-Poisson distribut
    ion, where the weight of the Beta-Poisson mean value is given a weight o
    f (1 - intrinsic.noise).
    The value should be a numeric between 0 and 1.