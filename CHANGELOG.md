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

# 2.1.0.6
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
