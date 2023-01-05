import numpy as np
data1 = np.loadtxt("scMultiSim_RNA_counts.csv", delimiter=",", skiprows=1)
data2 = np.loadtxt("scMultiSim_ATAC_seq.csv", delimiter=",", skiprows=1)
# JAMIE will assume matrices with the same number of rows are
# completely matched if not provided a correspondence matrix
corr = np.eye(data1.shape[0], data2.shape[0])


# # %%
# import matplotlib.pyplot as plt
# from jamie.evaluation import plot_regular

# # %%
# # Load cell-type labels
# data1 = np.loadtxt("C:/Dropbox/DaifengWangLab/JAMIE/examples/data/UnionCom/MMD/s1_mapped1.txt")
# data2 = np.loadtxt("C:/Dropbox/DaifengWangLab/JAMIE/examples/data/UnionCom/MMD/s1_mapped2.txt")
# type1 = np.loadtxt("C:/Dropbox/DaifengWangLab/JAMIE/examples/data/UnionCom/MMD/s1_type1.txt").astype(np.int)
# type2 = np.loadtxt("C:/Dropbox/DaifengWangLab/JAMIE/examples/data/UnionCom/MMD/s1_type2.txt").astype(np.int)
# type1 = np.array([f'Cell Type {i}' for i in type1])
# type2 = np.array([f'Cell Type {i}' for i in type2])

# # Visualize integrated latent spaces
# fig = plt.figure(figsize=(10, 5))
# plot_regular([data1, data2], [type1, type2], ['Modality 1', 'Modality 2'], legend=True)
# plt.tight_layout()
# plt.savefig('simulation_raw.png', dpi=300, bbox_inches='tight')

# %%
