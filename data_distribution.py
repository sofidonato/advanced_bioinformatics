#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas, numpy, seaborn
import scipy, scipy.signal
import matplotlib, matplotlib.pyplot
from scipy.stats import zscore
import seaborn as sns
import qnorm 
matplotlib.rcParams.update({'font.size':15, 'xtick.labelsize':30, 'ytick.labelsize':30, 'axes.labelsize':40, 'figure.figsize':(12, 8)})


# In[4]:


tpm = pandas.read_csv('/Users/ASUS/Documents/bioinf/tpm.csv', index_col=0)
print(tpm.shape)
tpm.head()


# In[5]:


#filter out genes that (almost) never show up   
expressed_genes = tpm[tpm.max(axis=1) >= 8]
expressed_genes.shape


# In[6]:


#bring data to log2 TPM +1
log2_tpm_PO = numpy.log2(expressed_genes + 1)
log2_tpm_PO


# In[55]:


'''HISTOGRAM'''

found_max = 8 # especifico el valor màxim aqui. El poso una mica mes alt
number_of_bins = found_max*10
print(number_of_bins)

absolute_max = 0  #defineixo absolute_max = 0
working_samples = log2_tpm_PO.columns.to_list()  #afegeixo les columnes amb les que vull treballar a una llista working_samples

most_likely_expressions = []
all_hats = []
for sample in working_samples:
    
    log2TPM = log2_tpm_PO.loc[:, sample] #selecciono una columna i la guardo a log2TPM
    if max(log2TPM) > absolute_max:   #si el valor maxim de log2TPM és major que l'absolut_max, absolut_max passa a ser aquest valor
        absolute_max = max(log2TPM)
       
    #print(numpy.min(log2TPM), numpy.max(log2TPM))
    
    # agafa el valor de dalt del mig de cada barra
    # agafa els 2 valors dels extrems de baix de cada barra i resta i divideix entre 2 per obtenir un unic valor de baix 
    hist, bin_edges = numpy.histogram(log2TPM, bins=number_of_bins, range=(0, found_max))
    half_bin = (bin_edges[1] - bin_edges[0])/2
    x = bin_edges + half_bin   
    x = x[:-1]   #elimina l'ultim valor pq no el necessitem
    
    #print(x)   #--> fem un 'print' per veure les dades i agafar el segon pic
    #print(hist)
    
    plotting_x = x#[1:500] #depen de com surti lhistograma agafem a pertir de unes dades o altres 
    plotting_hist = hist
    #print(plotting_x)
    
    #matplotlib.pyplot.plot(plotting_x, plotting_hist,'.', alpha=1/3)
    yhat = scipy.signal.savgol_filter(plotting_hist, 51, 3)
    matplotlib.pyplot.plot(plotting_x, yhat, '-', lw=2, alpha = 1/4)
    
    most_likely_expression = x[numpy.argmax(yhat)]
    most_likely_expressions.append(most_likely_expression)
    
    all_hats.append(yhat)
    
    
matplotlib.pyplot.xlim([numpy.min(plotting_x)+0.75, numpy.max(plotting_x)+0.25])
matplotlib.pyplot.ylim([0, 400])

matplotlib.pyplot.xlabel('log2 (TPM+1)')
matplotlib.pyplot.ylabel('Gene count')
matplotlib.pyplot.grid(ls=':')
#matplotlib.pyplot.legend(ncol=10, fontsize=12, bbox_to_anchor=(1.02, 1.25))

matplotlib.pyplot.tight_layout()

print(absolute_max)
    


# In[56]:


# Cálculo de la línea promedio
average_line = np.mean(all_hats, axis=0)

# Definición de los valores x para el gráfico
x_values = plotting_x

# Trazar la línea promedio
plt.plot(x_values, average_line, '-', lw=2, alpha=1/2, label='Promedio')

plt.xlim([np.min(x_values) + 0.75, np.max(x_values) + 0.25])
plt.ylim([0, 400])
plt.xlabel('log2 (TPM+1)')
plt.ylabel('Gene count')
plt.grid(ls=':')
plt.legend(ncol=10, fontsize=12, bbox_to_anchor=(1.02, 1.25))
plt.tight_layout()
plt.show()


# In[21]:


# We have a DataFrame called 'log2_tpm_PO' with gene expresion data and we want to distinguish the top_500 genes from the 
#other ones. Top 500 genes are the ones with higher mean and higher variance. 

# Calcula la media y la varianza de expresión para cada gen
average_expression = log2_tpm_PO.mean(axis=1)
variance_by_rows = log2_tpm_PO.var(axis=1)

# Calcula los puntajes Z para las columnas 'Average' y 'Variance'
average_zscores = zscore(average_expression)
variance_zscores = zscore(variance_by_rows)

# Calcula el puntaje de bondad sumando los puntajes Z de media y varianza
goodness_score = average_zscores + variance_zscores

# Crea un DataFrame con los puntajes de bondad y los nombres de los genes
genes_exp = pd.DataFrame({'Goodness_Score': goodness_score, 'Gene_Name': log2_tpm_PO.index})

# Selecciona los 500 genes con los puntajes de bondad más altos
top_500_genes = genes_exp.nlargest(500, 'Goodness_Score')

#Select the 50 top genes 
top_50_genes = genes_exp.nlargest(50, 'Goodness_Score')

#Select the 1000 top genes 
top_1000_genes = genes_exp.nlargest(1000, 'Goodness_Score')

#Select the 2000 top genes 
top_2000_genes = genes_exp.nlargest(2000, 'Goodness_Score')



# Muestra los primeros 5 genes en la lista de los mejores 500 genes
print(top_500_genes.head(5))


# In[17]:


#We use '.transpose()' function to change df format so we obtain values that were in columns now in rows and viceversa
transpose = log2_tpm_PO.transpose()
print(transpose.shape)
transpose.head()
transpose = transpose.dropna()
transpose


# In[24]:


# SCATTERPLOT WITH COEFICIENT OF VARIANCE

# Calcular la media y la desviación estándar para cada gen
media = transpose.mean()
std_dev = transpose.std()
cv = (std_dev / media) * 100  # Calcula el CV y multiplica por 100 para obtener un porcentaje

# Crear un DataFrame con las medias y desviaciones estándar
data_for_scatter = pd.DataFrame({'Media': media, 'Coeficient of variance': cv})

# Crear un scatter plot
plt.figure(figsize=(10, 8))

mida=10

# Crear un scatter plot con todos los genes en azul y forma circular
sns.scatterplot(x='Media', y='Coeficient of variance', data=data_for_scatter, color='blue', alpha=1/2, label='Otros Genes')

# Crear un scatter plot con los 500 mejores genes en rojo y forma cuadrada
sns.scatterplot(x='Media', y='Coeficient of variance', data=data_for_scatter.loc[top_500_genes.index], color='red', marker='D', alpha=1/2, label='Top 500 Genes')

# Personalizar el scatter plot
plt.title('Scatter Plot de Media vs. Coeficient of variance')
plt.xlabel('Media')
plt.ylabel('CV')

# Mostrar el scatter plot con una leyenda
plt.legend()

# Mostrar el scatter plot
plt.show()



# In[58]:


# NOW WE ARE GOING TO SAVE IN DIFFERENT DATAFRAMES AN SPECIFIC NUMBER OF TOP GENES WITHOUT NORMALIZATION TO SEE THE EFFECT 
#OF THE AMOUNT OF VARIABLES WE USE IN THE RESULTS. 

# Supongamos que 'top_genes' contiene los nombres de los genes que deseas mantener
important_50_genes = top_50_genes['Gene_Name'].tolist()

# Filtra las columnas de 'transpose' para mantener solo los genes presentes en 'top_genes'
important_50_dataframe = transpose[important_50_genes]
important_50_dataframe
important_50_dataframe.to_csv('/Users/ASUS/Documents/bioinf/wn_tpm_50.csv')


############################

important_500_genes = top_500_genes['Gene_Name'].tolist()

important_500_dataframe = transpose[important_500_genes]
important_500_dataframe
important_500_dataframe.to_csv('/Users/ASUS/Documents/bioinf/wn_tpm_500.csv')


######################
important_1000_genes = top_1000_genes['Gene_Name'].tolist()

important_1000_dataframe = transpose[important_1000_genes]
important_1000_dataframe.to_csv('/Users/ASUS/Documents/bioinf/wn_tpm_1000.csv')

#################
important_2000_genes = top_2000_genes['Gene_Name'].tolist()

important_2000_dataframe = transpose[important_2000_genes]
important_2000_dataframe.to_csv('/Users/ASUS/Documents/bioinf/wn_tpm_2000.csv')


# In[26]:


important_500_dataframe


# In[37]:


# Save the normalized dataframes in a csv file 

important_50_df= important_50_dataframe.T
important_50_df= qnorm.quantile_normalize(important_50_df, axis=1)
important_50_df.to_csv('/Users/ASUS/Documents/bioinf/tpm_50.csv')

important_500_df= important_500_dataframe.T
important_500_df= qnorm.quantile_normalize(important_500_df, axis=1)
important_500_df.to_csv('/Users/ASUS/Documents/bioinf/tpm_500.csv')

important_1000_df= important_1000_dataframe.T
important_1000_df= qnorm.quantile_normalize(important_1000_df, axis=1)
important_1000_df.to_csv('/Users/ASUS/Documents/bioinf/tpm_1000.csv')

important_2000_df= important_2000_dataframe.T
important_2000_df= qnorm.quantile_normalize(important_2000_df, axis=1)
important_2000_df.to_csv('/Users/ASUS/Documents/bioinf/tpm_2000.csv')



# In[38]:


## boxplot of 100 patients looking at 500 top genes  

# Transponer los datos normalizados
transposed_500 = important_500_dataframe[:100].T

# Crear un boxplot para cada paciente
plt.figure(figsize=(15, 8))
sns.boxplot(data=transposed_500, orient="v", palette="Set2")

# Personalizar el gráfico
plt.ylabel('log2 TPM+1')
plt.xlabel('Patients')
plt.title('Boxplot of Top 500 Gene Expression for Each Patient')

# Muestra el gráfico
plt.xticks([])  # Rotar las etiquetas del eje x para mayor legibilidad
plt.show()


# In[51]:


#Normalize transposed_500 dataframe, which contains 100 patients and 500 genes and save it in a csv file
q500 = qnorm.quantile_normalize(transposed_500, axis=1)
q500.to_csv('/Users/ASUS/Documents/bioinf/qtpm_500.csv')


# In[54]:


# Boxplot with normalized data

sns.boxplot(data=q500, orient="v", palette="Set2")
plt.ylabel('Z score')
plt.xlabel('Patients')
plt.title('Boxplot of Top 500 Gene Expression (Quantile Normalized) for Each Patient')
plt.xticks([])

