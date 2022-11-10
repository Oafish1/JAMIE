# BABEL Data Run on COLO320DM
The files needed are:
- `atac_rna_test_preds.h5ad`
- `rna_atac_test_preds.h5ad`
- `train_atac.h5ad`
- `train_rna.h5ad`
- `test_atac.h5ad`
- `test_rna.h5ad`

The `h5ad` files may be generated using [BABEL](https://github.com/wukevin/babel), installing as directed, and running
```python
python bin/train.py --data DM_rep4.h5
```

`DM_rep4.h5` can be found [here](https://office365stanford-my.sharepoint.com/:u:/g/personal/wukevin_stanford_edu/Edq1Cr6qejpOgzjZGa4bkvwB-LyH5MLbkLD6wGQCL4jvwA?e=T8IO54), or under "Reproducing pre-trained model" in the BABEL repository.

## Data source
[Colon Adenocarcinoma Cell Line](https://pubmed.ncbi.nlm.nih.gov/498117/)
