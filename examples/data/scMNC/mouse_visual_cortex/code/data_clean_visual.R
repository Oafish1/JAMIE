library(plyr)
library(reshape2)
library(stringr)
library(dplyr)

edata = read.csv("../data/efeature.csv",stringsAsFactors = F)
gdata = read.csv('../data/geneExp.csv',header=T,check.names = F)
meta = read.csv('../data/20200711_patchseq_metadata_mouse.csv',header=T,check.names = F,stringsAsFactors = F)
gdata_type = read.csv("../data/DER-21_Single_cell_markergenes_UMI.csv",stringsAsFactors = F,header=T)

edata$session_id = sapply(edata$ID,function(x){na.omit(unlist(strsplit(x, "[^0-9]+")))[3]})
edata$subject_id = sapply(edata$ID,function(x){na.omit(unlist(strsplit(x, "[^0-9]+")))[2]})
edata$session_idg = sapply(edata$session_id,FUN = function(x){meta$transcriptomics_sample_id[meta$ephys_session_id==x]})
meta$genotype = sapply(meta$t_type,function(x){na.omit(unlist(strsplit(x, " ")))[1]})
meta$dendrite_type = mapvalues(meta$dendrite_type,from = "sparsely spiny",to = "spiny")

# filter features
edata <- edata[,c(1:11,
                       12:18,21:23,#ramp
                       24:30,33:35,#long
                       36:42,45:47,#short
                       51)]
edata = na.omit(edata)

meta = meta[meta$t_type != "",]
meta = meta[meta$transcriptomics_sample_id %in% edata$session_idg,]
meta = meta[meta$genotype %in% c("Lamp5","Pvalb","Serpinf1","Sncg","Sst","Vip"),]
meta = meta[meta$dendrite_type == "aspiny",]

#filter elec
cellnames = meta$transcriptomics_sample_id
rownames(edata) = edata$session_idg
edata = edata[,-which(names(edata)=="session_idg")]
edata = edata[cellnames,]   

#filter gene
gdata = gdata[!duplicated(gdata$gene),]
rownames(gdata) = gdata$gene;gdata = gdata[,-1]
gdata_type$Cluster = sapply(gdata_type$Cluster,substr,start=1,stop=2)
genenames = unique(gdata_type$Gene[gdata_type$Cluster %in% c("Ex","In")])
genenames = na.omit(rownames(gdata)[match(genenames,toupper(rownames(gdata)))])
gdata = gdata[intersect(genenames,rownames(gdata)),cellnames]

#save to rda file
save(edata,gdata,meta,file = "../data/visual_data_filtered.rda")
