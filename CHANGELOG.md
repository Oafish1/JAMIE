# 4.3.3
- Small `sequential` argument fix
- SVG visualizations

# 4.3.2
- Added `return_statistic` to `plot_auroc...` function family
- Additional statistic reporting surrounding dropout
- Revised visualizations for loss clarity

# 4.3.1
- Additional runs for imputation
- Clarifications
- More options
- Small terminology fixes

# 4.3.0
- Add memory logging
- Added `hybrid` method to `plot_integrated`
- Added `plot_sample` for showing individual cell correlations
- Added `sort_type` option to `sort_by_interest` utility function
- Fixed ordering for `plot_distribution_alone`
- More runs, more data
- New visualization for `ATAC -> RNA` imputation gene importance
- Visualization fixes

# 4.2.3
- Figure changes
- Simulation data update

# 4.2.2
- Added loss logging
- Additional runs
 - Alternative phenotypes
 - Tuning
- Bugfixes
- More runs with partial correspondence
- Reorganization of notebooks
- Reruns
- Revised hashing function

# 4.2.1
- Reproducibility for UMAP

# 4.2.0
- Added `GNU GPL V3.0` License
- Better sampling during training
- Delete `generate_plot` class
- File structure revisions
- Raw ephys testing
- `README` updates
- Reproducibility for SHAP

# 4.1.8
- Import hierarchy fix
- `umap-learn` import fix
- `WR2MD` import versioning

# 4.1.7
- `README.md` updates
- Reruns

# 4.1.6
- Reruns
- Small `README.md` changes

# 4.1.5
- Change name from `ComManDo` to `JAMIE`
- Include data in repo
- `README.md` update

# 4.1.4
- Reruns
- Run leave-one-out on COLO320DM
- Visualization changes and reporting

# 4.1.3
- Reruns
- Small change to `evaluate_impact` background preview

# 4.1.2
- Reruns
- Visualization updates
- Weight standardization updates to reduce dimension dependence

# 4.1.1
- Added `feature_dict` argument to `plot_distribution_alone` for custom `xticks`
- Additional outlier protection for `plot_distribution_alone` visualizations
- Small fix with label formatting

# 4.1.0
- Add gradient clipping
- Added `plot_impact` to `evaluation` module
- Change early stop behavior and add `min_epochs`, mainly for KL annealing
- Fix scaling on `F` and `P` subsets during training
- Increase model robustness
- Lots of small fixes, errors mainly appeared with large datasets
- Model saving for all algorithms and outputs
- Reruns
- Visualization additions and changes

# 4.0.0
- Add `adjustText` for cleaner text notations
- Add `batch_step` option for typical AE iteration
- Add more SHAP visualizations
- Add outlier detection utility
- Add scDART
- Added auto-amending kwarg `pca_dim`
- Added BABEL datasets
- Adjusted evaluation figure text size
- Applied outlier detection to `plot_integrated`
- Change visualizations, especially for distributions
- Changed losses to VAE by default, `cosine`, `F`
- Changed sampling logic on distribution similarity calculation
- Fix bug for non-aligned datasets in `commando` module
- Fix SVD solver option in automated PCA
- Implement full VAE
- Implemented saving and loading models
- New interesting feature finding algorithm
- Stylistic changes in `evaluation`
- Various visualizations in `evaluation` module

# 3.8.2
- Fixed a bug concerning min-max normalization in the `compute_distances` function
- Implement SHAP and add visualizations
- Reruns

# 3.8.1
- Distribution similarity measure
- Evaluation style changes
- Reruns

# 3.8.0
- Add MMD-MA
- Add VAE functionality
- Add feature distribution previews
- Add inversion to model preprocessing
- Add trainable weighting to model aggregation function
- Additional error handling
- Additional visualization clarity
- Include `BrainChromatin` dataset
- Reruns
- Separate plots from `generate_figure` module
- Update evaluation module with imputed AUROC distribution
- Visualization reformatting

# 3.7.7
- Add configurable legend to `generate_figure`
- Added correlation and p-value to calibration plots
- Added `feature_names` argument to `generate_figure`
- Many Formatting changes for `generate_figure`
- Rename and format calibration plot
- Reruns

# 3.7.6
- Bugfix with `None` pca argument
- Reruns, small notebook formatting changes

# 3.7.5
- Added changeable `k` for `test_LabelTA` and integrated into `generate_figure`
 - Automatically chooses appropriate `k`
- Added optional `integrated_alg_shortnames` to `generate_figure`
- Fixed a bug where PCA was used on singular `None` modalities
- Fixed (subverted) a python bug which culls vars exclusively in lambda functions
 - Previously prevented proper usage of multiple `pca_dim`
- `generate_figure` formatting changes
- Reruns, including PCA fix for `scMNC-Visual`

# 3.7.4
- `generate_figure` add multi-column colors
- `generate_figure` formatting, title, layout changes
- Reruns
- Small code formatting change

# 3.7.3
- Add more comparison methods to notebooks
- Change `generate_figure` default layout
- Change optimizer for main `JAMIE` model
- Change plotting style for `generate_figure` stats
- Reruns

# 3.7.2
- Add custom colors to `generate_figure`
- Add metric heatmap to `generate_figure`
- Fixes to dimensions and generalization in `generate_figure`
- Formatting changes to default `generate_figure`
- Reruns on all `general_analysis` notebooks

# 3.7.1
- Rerun `scGEM` data

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
