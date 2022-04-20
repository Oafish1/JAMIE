# 3.7.0
- Add experimental `corr_method`
- Added `dist_method` choice for `sim_dist_func` which takes `cosine`, `euclidean`
- Generalize `F` representation to allow for negatives
 - Fixes problem with negative `sim`, `diff`, `F`, or `P`
- Reruns
- Use euclidean distance for `sim_dist_func`

# 3.6.1
- Added plug-in pre-calculated `F` matrix through `match_result`
- Allow for per-dataset `pca_dim`
- Fixed `preprocessing` variable default in `model`
- Test patch-seq dataset

# 3.6.0
- Add pca before use on full model
- Run on MMD-MA simulation data

# 3.5.4
- Add Davies-Bouldin Index
- Experimental losses
- Run on motor data
- Use `PF_Ratio` in runs

# 3.5.3
- Fix `scMNC` data loading
- `generate_figure`
  - Add `_get_..._shape()` functions
  - Add `_group_plot()` function for automated group partitioning
  - Add auto-selection of latent features for reconstruction visualization
  - Extend shape implementation
  - General reorganization
  - Streamlined main function
  - `height_ratio` fixes
- Remove mapping timer printing

# 3.5.2
- Added group partitioning functionality to `generate_figure`
- Refactored `generate_figure` to a class object

# 3.5.1
- Added custom model functionality with `model_class` argument
- Added simple model weight visualization to `generate_figure`
- Reruns on various levels of alignment
- Revised `generate_figure` formatting and standardized module format

# 3.5.0
- Added `integrated_use_pca` option for `generate_figure`
- Experimented with new losses
- New scaling on each loss variable
- Notebook preprocessing changes
- Reran notebooks
- Revised sim-dist measure to only include positives
- Revised model structure
- Stepping with epochs now rather than batches

# 3.4.0.0
- Mute output on various functions
- Add tuning function `utilities.tune_cm()`
- Add early stopping
- Add experimental loss parameters
- Add alternative similarity measures
- Add correlation visualization in `generate_figure`
- Revise `loss_weights` parameter
- Revise loss bookkeeping
- Fix F normalization
- Fix model normalization issue
- Remove KNN usage in main model
- Reruns with higher construction losses

# 3.4.0
- `generate_figure`
 - Added 3D plotting capability
 - Various formatting changes and style options
 - Silhouette coefficient visualizations
 - More modality prediction comparisons
- Add MMD-MA comparison
- Add scMNC data
- Reruns

# 3.3.0
- Added `generate_figure` to more concisely show results
- Small changes in several algorithms to mute output
- Renamed `joint_embedding` folder to `general_analysis`
- Re-ran notebooks

# 3.2.0
- Added `test_label_dist` to show inter-cell distance
- Re-run and revise notebooks, especially for `modality prediction`

# 3.1.0
- `aligned_idx` is now `P`, a matrix filled with priors
- Combined `P` and `F` matrices into aggregate `corr`
- Changes improved unaligned performance significantly
- Slightly reduced aligned performance
- Re-ran joint embedding notebooks
- Fixed bug in `knn` calculation
- Added `perfect_alignment` toggle for separate knn graph calculation method
- Cleaned up no-longer-used files

# 3.0.0
- Add evaluation graph for alignment assumptions
- Merge notebooks
- Example directory reorganization
- Removed certain errant checks
- Add modality prediction samples

# 2.1.0.9
- Added compatibility for partially aligned datasets using overlapping average vectors
- Added compatibility for differently-sized datasets
- Added "mix-in" metrics to control how much training is done on aligned sets
- Added more visualization for differently aligned datasets

# 2.1.0.8
- Make loss function more modular
  - Add switchable distance function (Euclidean, Manhattan, Cosine, etc.)
- Simplify loss function
- Fix similarity function
- Re-add connected KNN to F
- Notebook reruns

# 2.1.0.7
- Added inverse cross loss
- Notebook reruns

# 2.1.0.6 (3.0.0.0)
- Implement encoder-decoder model
- Add custom `model` module
- Re-run notebooks (doing well!)

# 2.1.0.5
- Removed temporary `neighborhood` module
- Added neighbor graph utility `knn` to `nn_funcs` module
- Added guaranteed connectivity to `knn`
- Tuning and reruns on all notebooks

# 2.1.0.4
- Modified `visualize` UnionCom function
- Cleaned loss handling in `project_nlma`
- More robust loss output
- Notebook rerun

# 2.1.0.3
- Implemented `test_closer`, measuring fraction of samples closer to true match
- Moved auxiliary loss calculations to new `nn_funcs` module
- Revised loss function
 - Refactored code
 - Added matrix versions of UC term and NLMA
 - Added naïve implementation of Gromov-Wasserstein distance
- Renamed and added notebooks `scGEM` and `MMD-MA`
- Removed `comparison.ipynb` and added comparisons in each notebook
- Reran notebooks

# 2.1.0.2
- Added `ALL FIRST` and `BATCH FIRST` calculation modes to hybrid `project_nlma`
- Renamed default NN step timing to `BATCH FIRST`
- Finished vanilla loss run with improved label transfer accuracy (`comparison_no-hybrid.ipynb`)
- Renamed and reran unfinished hybrid loss run (`sample.ipynb`)

# 2.1.0.1
- Deprecated existing `project_nlma`
- New `project_nlma` on `tsne` backend
 - Hybrid loss function
- Unfinished test runs
- Temporarily added partial `neighborhood` module from ManiNetCluster

# 2.1.0.0
- Remove two-step, gradient optimizations temporarily
- Comparison notebook
- Fix NLMA scaling (coefficient fixes)
- Re-added `tsne` projection method

# 2.0.5
- Small `construct_sparse` fix

# 2.0.4
- Cleanup, notes, file removal

# 2.0.3
- CPU fixes
- Large matrix exclusion fixes
- Expand/shrink matrix normalization
- Reruns
- Notes on coefficient/F-combination difficulties

# 2.0.2
- Small fixes
- Reruns

# 2.0.1
- AnnData input support
- Cell cycle partitioning support
- Notebook reruns

# 2.0.0
- Two-step optimization
 - `two_step_include_large`
 - Redundant calculation
- NLMA projection fixes
- GPU compatibility
- Sparse compatibility
- Utility functions

# 1.0.0
- Initial release
