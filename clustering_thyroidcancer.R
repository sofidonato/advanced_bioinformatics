library(readr)
#install.packages("NbClust")
library(NbClust)
#install.packages("factoextra")
library(factoextra)
#install.packages("umap")
library(umap)
library(ggplot2)
#install.packages("data.table")
library(data.table)

setwd("C:/Users/ASUS/Documents/bioinf")

# Cargar el archivo CSV sin asignar índices
expression <- read.csv("wdf.csv", header = TRUE)



dim(expression)
expression <- expression[1:500,2:521] #elimino la 1a columna
rownames(expression) <- expression$gene_id

t_expression <- t(expression)

rownames(t_expression) <- colnames(expression)
colnames(t_expression) <- rownames(expression)

pca_result <- prcomp(expression, rank. = 2, scale.= TRUE)
# 
# 
#my_biplot<- biplot(pca_result, labels=FALSE)
plot(pca_result$x[,1], pca_result$x[,2])



# BEST CLUSTERING 

umap_result <- umap(t_expression, n_components = 2)

nb <- NbClust(data = umap_result$layout, distance = "euclidean", min.nc = 3, 
              max.nc = 8, method = "ward.D2", index = "all") 

bclust_labels <- nb$Best.partition
table(bclust_labels)

umap_df <- data.frame(x = umap_result$layout[, 1], y = umap_result$layout[, 2], cluster = as.factor(bclust_labels))

shapes = c(0, 1, 2, 4, 5, 6)
cols = c('#A71B4B', '#EE850E', '#FAC763', '#C4F1AF', '#00A6B6', '#584B9F'
)

ggplot(umap_df, aes(x=x, y=y, color=cluster, shape=cluster)) + 
  xlab('UMAP dimension 1') + ylab('UMAP dimension 2') + 
  geom_point(alpha=.7, size=2) +
  scale_shape_manual(values = shapes) +
  scale_color_manual(values = cols) +
  ggtitle('UMAP projection of samples by clusters') +
  theme_light()

#guardo en un fitxer a quin cluster pertany cada pacient
write.csv(bclust_labels, "cluster_label.csv", row.names = TRUE)

