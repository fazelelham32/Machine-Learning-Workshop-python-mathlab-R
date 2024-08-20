#!/usr/bin/env python
# coding: utf-8

# 
# 
# <p align="right">In this exercise, we want to design a category with the help of our knowledge of univariate normal distribution</p>
# 
# <p align="right">Use the scikitlearn library to implement the models</p>
# 
# <p align="right">https://scikit-learn.org/stable/</p>

# # Read data

# <p align="right">First, we read the data from the library and then divide the data into two categories, training and testing</p>

# In[ ]:


from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# store the feature matrix (X) and response vector (y) 
X = iris.data
y = iris.target
print("our dataset has " + str(X.shape[1]) + " features. for more information about data surf the web")
# splitting X and y into training and testing sets
#you can change the test size, fit model with more or less data and see results
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 


# ### q1) print the number of train and test data and number of classes

# In[1]:


print("the number of train data is : " )
print("the number of test data is : " )
print("there is " + str() + " different classes in the dataset")


# # Training a model assuming Gaussian data distributionا

# <p align="right">Suppose we know that our data follows a Gaussian distribution and also that the features are independent of each other. Please teach a model for classifying classes with the help of the assumptions of the problem and without the help of ready-made models. Be careful that you should not use ready-made codes and you are only allowed to use ready-made functions for basic formulas such as average or variance.</p>

# In[ ]:


# training the model on training set 

# write your code here :


# <p align="right">Now measure the accuracy of your model with the help of test data</p>

# In[ ]:


#write your code here :


# # Training the model without knowing the data distribution

# <p align="right">Classify the data with the help of SVM classifier as well as a simple neural network and compare the accuracy with the previous section</p>

# In[ ]:


from sklearn import svm
#train svm model
#write yor code here :


# In[ ]:


from sklearn.neural_network import MLPClassifier
#use two hidden layers 
#train multi layer perceptron
#write yor code here :


# 
# 
# ---
# 
# 

# # Read new data

# In[ ]:


wine = datasets.load_wine()
# store the feature matrix (X) and response vector (y) 
X = wine.data 
y = wine.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 


# # Training a model assuming Gaussian distribution of data

# <p align="right">Suppose we know that our data follows a Gaussian distribution. With the help of ready-made libraries, train a model for classifying classes. You can also use ready-made libraries for this part, and there is no need to implement.</p>

# In[ ]:


from sklearn.naive_bayes import GaussianNB
# training the model on training set 

# write your code here :


# <p align="right">Now measure the accuracy of your model with the help of test data</p>

# In[ ]:


#write your code here :


# # Training the model without knowing the data distribution

# <p align="right">Classify the data with the help of SVM classifier as well as a simple neural network and compare the accuracy with the previous section</p>

# In[ ]:


#train svm model
#write yor code here :


# In[ ]:


#train multi layer perceptron
#write yor code here


# In[1]:


get_ipython().system('pip install nbconvert')


# 
# 
# ---
# 
# 

# In[2]:



get_ipython().system('jupyter nbconvert --to script --output "C:\\Users\\dor_mah\\Music\\importent folder\\main folder\\context projects\\ML\\codes and files\\project1"')


# In[ ]:




