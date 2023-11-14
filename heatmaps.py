#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries

import pandas as pd
import pandas, seaborn, numpy
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Read the files we are going to need 

important_50 = pandas.read_csv('/Users/ASUS/Documents/bioinf/wn_tpm_50.csv', index_col=0)
important_500 = pandas.read_csv('/Users/ASUS/Documents/bioinf/wn_tpm_500.csv', index_col=0)
important_1000 = pandas.read_csv('/Users/ASUS/Documents/bioinf/wn_tpm_1000.csv', index_col=0)
important_2000 = pandas.read_csv('/Users/ASUS/Documents/bioinf/wn_tpm_2000.csv', index_col=0)


# In[3]:


# 50 TOP GENES HEATMAP


scaler = StandardScaler() 
# Escalar los datos
scaled_50 = pd.DataFrame(scaler.fit_transform(important_50), columns=important_50.columns)

scaled_50=scaled_50.T

linkage_method = 'complete'
distance_metric = 'cosine'

sns.set(font_scale=0.9)
sns.clustermap(
    scaled_50,  # Tus datos normalizados
    cmap='bwr',   # Mapa de colores
    col_cluster=True,  # Realizar clustering en el eje de las columnas
    row_cluster=True,  # Realizar clustering en el eje de las filas
    vmin=-1,  # Valor mínimo en el mapa de colores
    vmax=1,   # Valor máximo en el mapa de colores
    method=linkage_method,  # Método de enlace para el clustering
    metric=distance_metric,  # Métrica de distancia
    yticklabels=[],  # Mostrar etiquetas en el eje y
    xticklabels=[],  # Etiquetas en el eje x
    cbar_kws= {'label': 'Z score'}  # Etiqueta para la barra de color

)



plt.title('{} {}'.format(linkage_method, distance_metric))
plt.tight_layout()
plt.show()


# In[5]:


# 2000 TOP GENES HEATMAP


scaler = StandardScaler() 
# Escalar los datos
scaled_2000 = pd.DataFrame(scaler.fit_transform(important_2000), columns=important_2000.columns)

scaled_2000=scaled_2000.T

linkage_method = 'complete'
distance_metric = 'cosine'

sns.set(font_scale=0.9)
sns.clustermap(
    scaled_2000,  # Tus datos normalizados
    cmap='bwr',   # Mapa de colores
    col_cluster=True,  # Realizar clustering en el eje de las columnas
    row_cluster=True,  # Realizar clustering en el eje de las filas
    vmin=-1,  # Valor mínimo en el mapa de colores
    vmax=1,   # Valor máximo en el mapa de colores
    method=linkage_method,  # Método de enlace para el clustering
    metric=distance_metric,  # Métrica de distancia
    yticklabels=[],  # Mostrar etiquetas en el eje y
    xticklabels=[],  # Etiquetas en el eje x
    cbar_kws= {'label': 'Z score'}
)



plt.title('{} {}'.format(linkage_method, distance_metric))
plt.tight_layout()
plt.show()


# In[6]:


# 1000 TOP GENES HEATMAP


scaler = StandardScaler() 
# Escalar los datos
scaled_1000 = pd.DataFrame(scaler.fit_transform(important_1000), columns=important_1000.columns)

scaled_1000=scaled_1000.T

linkage_method = 'complete'
distance_metric = 'cosine'

sns.set(font_scale=0.9)
sns.clustermap(
    scaled_1000,  # Tus datos normalizados
    cmap='bwr',   # Mapa de colores
    col_cluster=True,  # Realizar clustering en el eje de las columnas
    row_cluster=True,  # Realizar clustering en el eje de las filas
    vmin=-1,  # Valor mínimo en el mapa de colores
    vmax=1,   # Valor máximo en el mapa de colores
    method=linkage_method,  # Método de enlace para el clustering
    metric=distance_metric,  # Métrica de distancia
    yticklabels=[],  # Mostrar etiquetas en el eje y
    xticklabels=[],  # Etiquetas en el eje x
    cbar_kws= {'label': 'Z score'}  # Etiqueta para la barra de color

)



plt.title('{} {}'.format(linkage_method, distance_metric))
plt.tight_layout()
plt.show()


# In[7]:


#500 TOP GENES with linkage_method a "complete" y distance_metric a "cosine".

scaled_500 = pd.DataFrame(scaler.fit_transform(important_500), columns=important_500.columns)

sacled_500 = scaled_500.T

linkage_method = 'complete'
distance_metric = 'cosine'

sns.set(font_scale=0.9)
sns.clustermap(
    scaled_500,  # Tus datos normalizados
    cmap='bwr',   # Mapa de colores
    col_cluster=True,  # Realizar clustering en el eje de las columnas
    row_cluster=True,  # Realizar clustering en el eje de las filas
    vmin=-1,  # Valor mínimo en el mapa de colores
    vmax=1,   # Valor máximo en el mapa de colores
    method=linkage_method,  # Método de enlace para el clustering
    metric=distance_metric,  # Métrica de distancia
    yticklabels=[],  # Mostrar etiquetas en el eje y
    xticklabels=[],  # Etiquetas en el eje x
    cbar_kws= {'label': 'Z score'}  # Etiqueta para la barra de color

)



plt.title('{} {}'.format(linkage_method, distance_metric))
plt.tight_layout()
plt.show()


# In[8]:


# TOP 500 HEATMAP with linkage_method = "ward" y distance_metric = "euclidean".

scaled_500 = pd.DataFrame(scaler.fit_transform(important_500), columns=important_500.columns)

sacled_500 = scaled_500.T

linkage_method = 'ward'
distance_metric = 'euclidean'

sns.set(font_scale=0.9)
sns.clustermap(
    scaled_500,  # Tus datos normalizados
    cmap='bwr',   # Mapa de colores
    col_cluster=True,  # Realizar clustering en el eje de las columnas
    row_cluster=True,  # Realizar clustering en el eje de las filas
    vmin=-1,  # Valor mínimo en el mapa de colores
    vmax=1,   # Valor máximo en el mapa de colores
    method=linkage_method,  # Método de enlace para el clustering
    metric=distance_metric,  # Métrica de distancia
    yticklabels=[],  # Mostrar etiquetas en el eje y
    xticklabels=[],  # Etiquetas en el eje x
    cbar_kws= {'label': 'Z score'}  # Etiqueta para la barra de color

)



plt.title('{} {}'.format(linkage_method, distance_metric))
plt.tight_layout()
plt.show()


# In[9]:


# TOP 500 GENES using linkage_method = "avarage" y distance_metric = "cityblock".

scaled_500 = pd.DataFrame(scaler.fit_transform(important_500), columns=important_500.columns)

scaled_500 = scaled_500.T

linkage_method = 'average'
distance_metric = 'cityblock'

sns.set(font_scale=0.9)
sns.clustermap(
    scaled_500,  # Tus datos normalizados
    cmap='bwr',   # Mapa de colores
    col_cluster=True,  # Realizar clustering en el eje de las columnas
    row_cluster=True,  # Realizar clustering en el eje de las filas
    vmin=-1,  # Valor mínimo en el mapa de colores
    vmax=1,   # Valor máximo en el mapa de colores
    method=linkage_method,  # Método de enlace para el clustering
    metric=distance_metric,  # Métrica de distancia
    yticklabels=[],  # Mostrar etiquetas en el eje y
    xticklabels=[],  # Etiquetas en el eje x
    cbar_kws= {'label': 'Z score'}  # Etiqueta para la barra de color

)



plt.title('{} {}'.format(linkage_method, distance_metric))
plt.tight_layout()
plt.show()


# In[ ]:




