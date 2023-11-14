#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import requests
import re
import pandas as pd 


# In[3]:


# Open GDC manifest file.
mf_name = input('Write manifest file name: ') 
data = pd.read_csv(mf_name,delimiter='\t')

# Directory where the files are saved.  
output_directory = "/Users/ASUS/Documents/bioinf/data"
os.makedirs(output_directory, exist_ok=True)

count = 0
for _, row in data.iterrows():
    # Loop through the file IDs and download each file.
    file_id = str(row[0])  
    data_endpt = "https://api.gdc.cancer.gov/data/{}".format(file_id)
    response = requests.get(data_endpt, headers={"Content-Type": "application/json"})

    # Check if the request was successful before proceeding.
    if response.status_code == 200:
        # The file name can be found in the header within the Content-Disposition key.
        response_head_cd = response.headers.get("Content-Disposition")
        if response_head_cd:
            file_name = re.findall("filename=(.+)", response_head_cd)[0]

            # Join the directory path with the file name to create the complete file path.
            output_file_path = os.path.join(output_directory, file_name)

            with open(output_file_path, "wb") as output_file:
                output_file.write(response.content)
                count = count + 1
                print(f"Downloaded file nº {count}: {file_name} " )
        else:
            print(f"Failed to get file name for file ID: {file_id}")
    else:
        print(f"Failed to download file ID: {file_id} - Status Code: {response.status_code}")


# In[ ]:


# Get the list of all files and directories
# in the root directory
path = "/Users/ASUS/Documents/bioinf/data"
dir_list = os.listdir(path)
  
print("Files and directories in '", path, "' :") 
  


# In[ ]:


concat_df=pandas.DataFrame()
print(type(concat_df))
for element in dir_list:#[:100]: 
    #We first visualize the name of the file 
    #print(element)

    #specify path
    path_to_data="C:/Users/ASUS/Documents/bioinf/data/"
    full_path=path_to_data+element
    

    #Create a dataframe of that file
    df = pandas.read_csv(full_path, sep="\t",skiprows=1, index_col=0)
    df.drop(['gene_name', 'gene_type'], axis=1, inplace = True)
    df.drop(['N_unmapped', 'N_multimapping', 'N_noFeature', 'N_ambiguous'], axis=0, inplace= True)
    
    #specify row and column indexes
    #file_id = str(row[1]).split('.')[0]
    
    #take the first column of each one
    expression_column = df['tpm_unstranded']
    expression_column = expression_column.rename(element.split('.')[0])   ######################
    
    if concat_df.shape == (0,0):
        concat_df = expression_column 
        print('starting to construct the concatenated dataframe')
    else:
        concat_df = pandas.concat([concat_df, expression_column], axis=1)


# In[ ]:


concat_df


# In[ ]:


#convert df to csv.
concat_df.to_csv('/Users/ASUS/Documents/bioinf/tpm.csv')

