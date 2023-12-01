#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importem les llibreries necessàries
import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler



# In[2]:


#Llegim el fitxer csv on hi tenim els clusters corresponents a cada pacient
archivo = "C:/Users/ASUS/Documents/bioinf/kmeans_labels.csv"
cluster_labels = pd.read_csv(archivo, index_col = 0)
cluster_labels 


# In[3]:


# Llegim el fitxer csv on hi tenim pacients i gens
archiu = "C:/Users/ASUS/Documents/bioinf/wdf.csv"
wdf = pd.read_csv(archiu, index_col=0)
wdf


# In[39]:


# Initialize the StandardScaler
scaler = StandardScaler()

#Scale the data
scaler = StandardScaler().fit(wdf)
scaled_data = scaler.transform(wdf)


# Convert the scaled data back to a DataFrame (if needed)
wdf_scaled = pd.DataFrame(scaled_data, columns=wdf.columns, index=wdf.index)
# Now wdf_scaled contains the scaled data

wdf_scaled.to_csv('/Users/ASUS/Documents/bioinf/scaled_data.csv')


# In[37]:


print(wdf_scaled.mean(axis = 0))
print(wdf_scaled.std(axis = 0))


# In[38]:


#Fem la transposada del dataframe wdf 
wdf_transposed = wdf_scaled.transpose()

wdf_transposed


# In[26]:


# passem els dos dataframes a array
X = wdf_transposed.to_numpy()
y = cluster_labels.to_numpy()
X


# In[27]:


#Dividim les dades en conjunts d'entrenament i prova, tenint una mostra de test del 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


#Creació i entrenament del model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluació del model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Exactitud del modelo: {accuracy}')
print(f'Reporte de clasificación:\n{report}')


# In[24]:


y_list = y.tolist()


# In[28]:


# Classification Tree

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load example dataset (Iris dataset)
#iris = load_iris()
X = wdf.to_numpy()
X_t=X.T

gene_names=wdf.index.tolist() #this is the y

# Initialize DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Fit the classifier on the data
clf.fit(X_t, y)

# Plot the tree
plt.figure(figsize=(13, 8))
plot_tree(clf, filled=True, feature_names=gene_names, class_names=[str(label) for label in y_list], fontsize=6)
plt.show()
plt.savefig(f"/Users/ASUS/Documents/bioinf/tree.svg")


# In[ ]:


# ROC CURVE FOR EACH CLUSTER 

# Obtener datos
X = wdf.to_numpy()
X_transposed = X.T

archivo_csv = "C:/Users/ASUS/Documents/bioinf/kmeans_labels.csv"

cluster_lab_df = pd.read_csv(archivo_csv)
cluster_labels = cluster_lab_df.iloc[:, 1].values
cluster_lab_df.columns = ['patient', 'cluster']


# In[ ]:


y = cluster_labels


# In[ ]:


# Split de datos
X_train, X_test, y_train, y_test = train_test_split(X_transposed, y, test_size=0.2, random_state=42)


# In[ ]:


# Modelo
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)


# In[ ]:


# Binarizar etiquetas
y_bin = label_binarize(y_test, classes=list(set(y)))


# In[ ]:


# Inicializar gráfico
plt.figure(figsize=(8, 6))

# Calcular la curva ROC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()
interp_tpr = dict()  # Diccionario para almacenar valores interpolados

for i in range(len(set(y))):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Interpolar para obtener una curva más suave
    interp_tpr[i] = np.interp(np.linspace(0, 1, 100), fpr[i], tpr[i])

# Calcular el micro promedio de ROC-AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Interpolar para el micro-average
interp_tpr["micro"] = np.interp(np.linspace(0, 1, 100), fpr["micro"], tpr["micro"])

# Dibujar la curva ROC para cada clase
for i in range(len(set(y))):
    plt.plot(np.linspace(0, 1, 100), interp_tpr[i], label=f'Cluster {i} (AUC = {roc_auc[i]:.2f})')

# Dibujar la curva ROC micro promedio
plt.plot(np.linspace(0, 1, 100), interp_tpr["micro"],
         label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle='--')

# Configuración del gráfico
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('Tasa de falso positivo (FPR)')
plt.ylabel('Tasa de verdadero positivo (TPR)')
plt.title('Curva ROC para cada cluster')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# FEATURE IMPORTANCE

# Obtener importancia de las características
importances = model.feature_importances_

# Obtener los índices de las características más importantes
top_gene_indices = importances.argsort()[::-1][:10]

# Obtener los nombres de los genes correspondientes del índice del DataFrame
top_genes = wdf.index[top_gene_indices]

print("Top 10 genes:")
print(top_genes)


# In[ ]:


#Visualitzem el model en gràfica
feature_importances = model.feature_importances_
#Creem un bar chart per tal de visualitzar amb feature importances
matplotlib.pyplot.figure(figsize=(8, 6))
matplotlib.pyplot.bar(range(len(feature_importances)), feature_importances)
matplotlib.pyplot.xlabel('Feature Index')
matplotlib.pyplot.ylabel('Feature Importance')
matplotlib.pyplot.title('Random Forest Feature Importances')

