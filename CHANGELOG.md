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
