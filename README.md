# ComManDo
`ComManDo` is a modification of `UnionCom` to utilize non-linear manifold alignment in its final projection step.  Various optimizations are also included.  Currently, functionality in standard methods (`tsne`, etc.) is limited but will be added back at a later date.

## Projection Method
[NLMA](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6329-2) has been added as an alternative to other projection methods in `UnionCom` such as `tsne` through the use of the calculated `F` matrix as a correlation matrix.  Our hope is that this will allow for greater data retention.

## Optimizations
### Gradient Optimization
Through random aggregation of cell data, gradients can be computed faster and cast back out to the original data sources, allowing for a generated `F` matrix with no lost resolution while cutting computation time.  With this method, however, criterion for convergence becomes more strict.

### Two-Step Aggregation
Cells are grouped into pseudocells.  Inter and intra-psuedocell `F` matrices are calculated, resulting in much faster (~20x) computation time and much less memory usage.  Currently, the whole matrix needs to be reconstructed before calculating eigenvectors, however, only delaying the memory requirement.  Additionally, the bulk of computation time is taken up by eigenvector calculation.
