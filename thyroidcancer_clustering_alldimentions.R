
library(readr)
library(NbClust)
library(factoextra)
library(umap)
library(ggplot2)
library(data.table)
library(svglite)

setwd("C:/Users/ASUS/Documents/bioinf")

# Cargar el archivo CSV sin asignar índices
expression <- read.csv("wdf.csv", header = TRUE)


dim(expression)
expression <- expression[1:500,2:521] #elimino la 1a columna
rownames(expression) <- expression$gene_id

t_expression <- t(expression)

rownames(t_expression) <- colnames(expression)
colnames(t_expression) <- rownames(expression)


nb <- NbClust(data = t_expression, distance = "euclidean", min.nc = 3, 
              max.nc = 7, method = "complete", index = "all") 

# Realizar clustering con K-means en datos de alta dimensión
set.seed(123)  # Establecer semilla para reproducibilidad

# Ejemplo de clustering con K-means usando 5 clusters (puedes ajustar el número)
num_clusters <- 5
kmeans_result <- kmeans(t_expression, centers = num_clusters)

# Obtener etiquetas de los clusters asignados a cada muestra
kmeans_labels <- kmeans_result$cluster

# Mostrar la distribución de muestras en cada cluster
table(kmeans_labels)

# Explore the results
kmeans_labels <- kmeans_result$cluster
table(kmeans_labels)


# Realizar PCA para reducción de dimensionalidad
pca_result <- prcomp(t_expression, scale. = TRUE)

# Obtener las coordenadas de las dos primeras componentes principales
pca_coordinates <- as.data.frame(pca_result$x[, 1:2])

# Agregar las etiquetas de cluster al dataframe resultante
pca_coordinates$cluster <- as.factor(kmeans_labels)

# Graficar los datos en un scatterplot con colores correspondientes a los clusters
library(ggplot2)

ggplot(pca_coordinates, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(size = 3, alpha = 0.6) +
  scale_color_discrete(name = "Cluster") +
  labs(x = "PC1", y = "PC2") +
  ggtitle("PCA plot of K-means clusters") +
  theme_minimal()

# Guardar las etiquetas de cluster en un archivo CSV
write.csv(kmeans_labels, "kmeans_labels.csv", row.names = TRUE)

