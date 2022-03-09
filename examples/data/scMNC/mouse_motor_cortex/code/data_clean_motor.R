library(readxl)
edata = read.csv("../data/motor_efeature.csv",header = T,stringsAsFactors = F)
gdata_exon = read.csv("../data/motor_gene_exon_counts.csv",header = T,stringsAsFactors = F)
gdata_intron = read.csv("../data/motor_gene_intron_counts.csv",header = T,stringsAsFactors = F)
colnames(gdata_exon)[2:ncol(gdata_exon)] = sapply(colnames(gdata_exon)[2:ncol(gdata_exon)],function(x){na.omit(unlist(strsplit(x, "X")))[2]})
colnames(gdata_intron)[2:ncol(gdata_intron)] = sapply(colnames(gdata_intron)[2:ncol(gdata_intron)],function(x){na.omit(unlist(strsplit(x, "X")))[2]})

#t-types
meta = read_excel("../data/motor_meta_data.xlsx")
cellnames = meta$Cell[meta$`RNA type` %in% names(table(meta$`RNA type`))]

#remove NA 
edata = na.omit(edata)
gdata_intron = gdata_intron[,colSums(is.na(gdata_intron)) == 0]
gdata_exon = gdata_exon[,colSums(is.na(gdata_exon)) == 0]

#merge gdata
gdata_intron = gdata_intron[,intersect(colnames(gdata_intron),colnames(gdata_exon))]
gdata_exon = gdata_exon[,intersect(colnames(gdata_intron),colnames(gdata_exon))]

#neuronal genes
gdata = rbind(gdata_exon,gdata_intron)
gdata = gdata[!duplicated(gdata[,1]),]
#write.csv(gdata,"../data/geneExp_motor.csv",row.names = F)
rownames(gdata) =gdata[,1];gdata =gdata[,-1]
rownames(edata) =edata[,1];edata =edata[,-1]
gdata_type = read.csv("../data/DER-21_Single_cell_markergenes_UMI.csv",stringsAsFactors = F,header=T)
gdata_type$Cluster = sapply(gdata_type$Cluster,substr,start=1,stop=2)
genenames = unique(gdata_type$Gene[gdata_type$Cluster %in% c("Ex","In")])
genenames = na.omit(rownames(gdata)[match(genenames,toupper(rownames(gdata)))])
cellnames = intersect(cellnames,colnames(gdata))
cellnames = intersect(cellnames,rownames(edata))
t_type = meta$`RNA family`[match(cellnames,meta$Cell)]
t_type_spec = meta$`RNA type`[match(cellnames,meta$Cell)]

# X,Y
edata = edata[cellnames,]
gdata = data.matrix(gdata[genenames,cellnames])
meta = meta[match(cellnames,meta$Cell),]

#save to rda file
save(edata,gdata,meta,file = "../data/motor_data_filtered.rda")
