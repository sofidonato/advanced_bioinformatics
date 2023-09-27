#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas


# In[3]:


# Get the list of all files and directories
# in the root directory
path = "/Users/ASUS/Documents/proves_bioinf"
dir_list = os.listdir(path)
  
print("Files and directories in '", path, "' :") 
  
# print the list
print(dir_list)





# In[4]:


#We use a for loop to create a list containing the names of the files in our directory

for element in dir_list:
    print(element)


# In[18]:


import pandas

concat_df=pandas.DataFrame()
print(type(concat_df))
for element in dir_list: 
    #We first visualize the name of the file 
    print(element)

    #specify path
    path_to_data="C:/Users/ASUS/Documents/proves_bioinf/"
    full_path=path_to_data+element
 
    #Create a dataframe of that file
    df = pandas.read_csv(full_path, sep="\t",skiprows=1)
    
    #take the first column of each one
    expression_column = df['tpm_unstranded']
    
    if concat_df.shape == (0,0):
        concat_df = expression_column   ############################################ put row indexes 
        print('starting to construct the concatenated dataframe')
    else:
        concat_df = pandas.concat([concat_df, expression_column], axis=1)


# In[8]:


concat_df

