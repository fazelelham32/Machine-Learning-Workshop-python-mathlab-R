#!/usr/bin/env python
# coding: utf-8

# There is a one-dimensional dataset, blue class dataset (C1) from Gaussian distribution with =0 and μ =0.1 and class σ
# Red (C2) follows a Gaussian distribution with =0.25 and μ =0.1. The number of data in each class σ
# It's the same.

# ## Derive the Bayes decision boundary equation

# Answer: The decision equation is obtained by combining the distributions of each class. By equating the above two distributions and solving the equation, the following value is obtained:
# x = 0.125

# It is placed in the attachment of the aforementioned documents. The first column is the attribute and the second column is its class.

# ## Draw the distribution function of two categories and the equation of the decision boundary and the data (data.csv) in a graph. Show the distribution function and the data of each class with different colors.

# In[ ]:





# ## Using the equation that you obtained, the (accuracy) of Bayes on the test data get (test.csv).

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


get_ipython().system('pip install nbconvert')


# In[3]:



get_ipython().system('jupyter nbconvert --to script --output "C:\\Users\\dor_mah\\Music\\importent folder\\main folder\\context projects\\ML\\codes and files\\project5.py" "project5.ipynb"')


# In[6]:


from IPython.display import Image


# In[14]:


Image(filename='C:/Users/dor_mah/Music/importent folder/main folder/context projects/ML/codes and files/image.jpg')


# In[15]:


dataset = pd.read_csv('C:\Users\dor_mah\Music\importent folder\main folder\context projects\ML\codes and files\data\data.csv')
test = pd.read_csv('test.csv')


# In[ ]:




