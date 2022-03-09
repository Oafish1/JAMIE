# #esssential packages
# if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
# BiocManager::version()
# BiocManager::install("SingleCellExperiment")
# # If your bioconductor version is previous to 4.0, see the section bellow
# 
# ## Required
# #BiocManager::install(c("AUCell", "RcisTarget"))
# #BiocManager::install(c("GENIE3")) # Optional. Can be replaced by GRNBoost
# if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
# devtools::install_github("aertslab/AUCell")
# devtools::install_github("aertslab/RcisTarget")
# devtools::install_github("aertslab/GENIE3")
# devtools::install_github("aertslab/SCENIC@v1.1.2")
# packageVersion("SCENIC")
# 
# ## Optional (but highly recommended):
# # To score the network on cells (i.e. run AUCell):
# BiocManager::install(c("zoo", "mixtools", "rbokeh"))
# # For various visualizations and perform t-SNEs:
# BiocManager::install(c("DT", "NMF", "ComplexHeatmap", "R2HTML", "Rtsne"))
# # To support paralell execution (not available in Windows):
# BiocManager::install(c("doMC", "doRNG"))
# 
# # To export/visualize in http://scope.aertslab.org
# devtools::install_github("aertslab/SCopeLoomR", build_vignettes = TRUE)

#mouse databease
dbFiles <- c("https://resources.aertslab.org/cistarget/databases/mus_musculus/mm9/refseq_r45/mc9nr/gene_based/mm9-500bp-upstream-7species.mc9nr.feather",
             "https://resources.aertslab.org/cistarget/databases/mus_musculus/mm9/refseq_r45/mc9nr/gene_based/mm9-tss-centered-10kb-7species.mc9nr.feather")
dir.create("cisTarget_databases"); setwd("cisTarget_databases") # if needed
for(featherURL in dbFiles)
{
  download.file(featherURL, destfile=basename(featherURL)) # saved in current dir
}
#get TF
library(RcisTarget)
geneMat = read.csv('data3/20200513_Mouse_PatchSeq_Release_count.v2.csv/20200513_Mouse_PatchSeq_Release_count.v2/20200513_Mouse_PatchSeq_Release_count.v2.csv',header=T,check.names = F)
geneSet = geneMat[,1]
geneLists = list(geneSetName=geneSet)
featherURL <- "https://resources.aertslab.org/cistarget/databases/mus_musculus/mm9/refseq_r45/mc9nr/gene_based/mm9-tss-centered-10kb-10species.mc9nr.feather" 
download.file(featherURL, destfile=basename(featherURL)) # saved in current dir
# Search space: 10k bp around TSS - Mouse
motifRankings <- importRankings("mm9-tss-centered-10kb-10species.mc9nr.feather")
# Load the annotation to human transcription factors
data(motifAnnotations_mgi)
motifAnnotations_mgi[199:202,]
TFgenes = data.frame(TF = intersect(motifAnnotations_mgi[,3],geneSet))
# write.csv(TFgenes,"../data/TFgenes.csv")




# SCENIC main analysis
suppressPackageStartupMessages({
  library(SCENIC)
  library(AUCell)
  library(RcisTarget)
  library(KernSmooth)
  library(BiocParallel)
  library(ggplot2)
  library(data.table)
  library(grid)
})

# Read data & filter
expMat = read.csv("../data/geneExp.csv")
NMA_result = read.csv("data/efeature_NMA.csv")
rownames(expMat) = expMat[,1];expMat =as.matrix(expMat[,-1])
expMat = expMat[,NMA_result$cellnames]

#cell info
nGene = apply(expMat,2,FUN = function(x){sum(x != 0)})
nUMI = apply(expMat,2,FUN = function(x){sum(x)})
CellType = NMA_result$gmm_cluster
cellInfo <- data.frame(CellType,nGene,nUMI)
saveRDS(cellInfo, file="int/cellInfo.Rds")

# yum install xorg-x11-server-Xvfb
# xvfb-run R script

cluster = 1

# SCENIC
suppressPackageStartupMessages({
  library(SCENIC)
  library(AUCell)
  library(RcisTarget)
  library(KernSmooth)
  library(BiocParallel)
  library(ggplot2)
  library(data.table)
  library(grid)
})

# Initialize SCENIC settings
org <- "mgi" # or hgnc, or dmel
dbDir <- "cisTarget_databases" # RcisTarget databases location
myDatasetTitle <- "SCENIC example on Mouse brain" # choose a name for your analysis
data(defaultDbNames)
dbs <- defaultDbNames[[org]]
scenicOptions <- initializeScenic(org=org, dbDir=dbDir, dbs=dbs, datasetTitle=myDatasetTitle, nCores=10) 
saveRDS(scenicOptions, file="int/scenicOptions.Rds") 

# Read data & filter
dbs <- defaultDbNames[["mgi"]]
expMat = read.csv("../data/geneExp.csv")
NMA_result = read.csv("../data/efeature_NMA.csv")
expMat = expMat[!duplicated(expMat[,1]),]
rownames(expMat) = expMat[,1];expMat =as.matrix(expMat[,-1])
expMat = expMat[,NMA_result$cellnames[NMA_result$gmm_cluster == cluster]]

# Gene/cell filter/selection
TF_genes = read.csv("../data/TFgenes.csv")
gdata_type = read.csv("../data/DER-21_Single_cell_markergenes_UMI.csv",stringsAsFactors = F,header=T)
gdata_type$Cluster = sapply(gdata_type$Cluster,substr,start=1,stop=2)
genenames = unique(gdata_type$Gene[gdata_type$Cluster %in% c("Ex","In")])
genesKept = union(genenames,TF_genes[,1])
genesKept = intersect(genesKept,rownames(expMat))
expMat_filtered <- expMat[genesKept, ]
dim(expMat_filtered)
expMat_filtered <- log2(expMat_filtered+1)

# Correlation
runCorrelation(expMat_filtered, scenicOptions)
# GENIE3
runGenie3(expMat_filtered, scenicOptions)

scenicOptions <- readRDS("int/scenicOptions.Rds")
scenicOptions@settings$verbose <- TRUE
scenicOptions@settings$nCores <- 10
scenicOptions@settings$dbs <- scenicOptions@settings$dbs["10kb"]

options(bitmapType = 'cairo')
runSCENIC_1_coexNetwork2modules(scenicOptions)
runSCENIC_2_createRegulons(scenicOptions, coexMethod=c("top5perTarget"))
runSCENIC_3_scoreCells(scenicOptions, expMat_filtered)
#saveRDS(scenicOptions, file="int/scenicOptions.Rds")