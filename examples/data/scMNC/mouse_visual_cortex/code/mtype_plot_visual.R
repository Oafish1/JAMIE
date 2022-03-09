library(plyr)
library(reshape2)
library(stringr)
library(dplyr)
library(ManiNetCluster)
library(ggplot2)
library(RColorBrewer)

Dim_red = function(edata,gdata,method,class,cellnames,d,k_NN,k_medoid){
  #e-feature
  X = apply(edata,2,scale)
  #g-feature
  Y=t(log10(gdata+1))
  #Dim_red
  XY_corr=Correspondence(matrix=diag(nrow(X)))
  df=ManiNetCluster(X,Y,nameX='Ephys',nameY='Expr',corr=XY_corr,d=d,
                    method=method,k_NN=k_NN,k_medoids=k_medoid)
  df$ttype = rep(class,2)
  df$cellnames = rep(unlist(cellnames),2)
  return(df[,-1])
}

edata = read.csv("../data/efeature.csv",stringsAsFactors = F)
gdata = read.csv('../data/geneExp.csv',header=T,check.names = F)
meta = read.csv('../data/20200711_patchseq_metadata_mouse.csv',header=T,check.names = F,stringsAsFactors = F)
gdata_type = read.csv("../data/DER-21_Single_cell_markergenes_UMI.csv",stringsAsFactors = F,header=T)

edata$session_id = sapply(edata$ID,function(x){na.omit(unlist(strsplit(x, "[^0-9]+")))[3]})
edata$subject_id = sapply(edata$ID,function(x){na.omit(unlist(strsplit(x, "[^0-9]+")))[2]})
edata$session_idg = sapply(edata$session_id,FUN = function(x){meta$transcriptomics_sample_id[meta$ephys_session_id==x]})
meta$genotype = sapply(meta$t_type,function(x){na.omit(unlist(strsplit(x, " ")))[1]})
meta$dendrite_type = mapvalues(meta$dendrite_type,from = "sparsely spiny",to = "spiny")

edata <- edata[,c(1:11,
                       12:18,21:23,#ramp
                       24:30,33:35,#long
                       36:42,45:47,#short
                       51)]
edata = na.omit(edata)

meta = meta[meta$transcriptomics_sample_id %in% edata$session_idg,]
meta = meta[meta$dendrite_type %in% c("spiny","aspiny"),]
meta = rbind(meta[meta$dendrite_type == "spiny",],
              meta[meta$genotype %in% c("Lamp5","Pvalb","Serpinf1","Sncg","Sst","Vip") & meta$dendrite_type == "aspiny",])

cellnames = meta$transcriptomics_sample_id
rownames(edata) = edata$session_idg
edata = edata[,-which(names(edata)=="session_idg")]
edata = edata[cellnames,]                               

gdata = gdata[!duplicated(gdata$gene),]
rownames(gdata) = gdata$gene;gdata = gdata[,-1]
gdata_type$Cluster = sapply(gdata_type$Cluster,substr,start=1,stop=2)
genenames = unique(gdata_type$Gene[gdata_type$Cluster %in% c("Ex","In")])
genenames = na.omit(rownames(gdata)[match(genenames,toupper(rownames(gdata)))])
gdata = gdata[intersect(genenames,rownames(gdata)),cellnames]

dendrite_type = sapply(cellnames,FUN = function(x){meta$dendrite_type[meta$transcriptomics_sample_id==x]})
n = nrow(edata)
method = c('linear manifold','cca','manifold warping','nonlinear manifold aln','nonlinear manifold warp')
#NMA
NMA_res = Dim_red(edata,gdata,method = method[4],class = dendrite_type, cellnames =cellnames,d=3L,k_NN=2L,k_medoid=5L)
NMA_res_e = NMA_res[NMA_res$data=="Ephys",]
NMA_res_t= NMA_res[NMA_res$data=="Expr",]
#Fig 2C
library(plot3D)
points3D(x=NMA_res_e$Val0, y=NMA_res_e$Val1, z=NMA_res_e$Val2,pch = 19,cex=0.5,bty="g",ticktype = "detailed", theta = 40, phi = 10,
         xlab = "",ylab = "",zlab = "",
         colvar = as.numeric(mapvalues(dendrite_type,names(table(dendrite_type)),1:2)),col =alpha(brewer.pal(6,"Spectral")[c(2,5)],0.8),
         colkey = F)
points3D(x=NMA_res_t$Val0, y=NMA_res_t$Val1, z=NMA_res_t$Val2,pch = 19,cex=0.5,bty="g",ticktype = "detailed", theta = 40, phi = 10,
         xlab = "",ylab = "",zlab = "",
         colvar = as.numeric(mapvalues(dendrite_type,names(table(dendrite_type)),1:2)),col =alpha(brewer.pal(6,"Spectral")[c(2,5)],0.8),
         colkey = F)
