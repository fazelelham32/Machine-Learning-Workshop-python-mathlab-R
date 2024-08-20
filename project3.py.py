#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nbconvert')


# In[2]:



get_ipython().system('jupyter nbconvert --to script --output "C:\\Users\\dor_mah\\Music\\importent folder\\main folder\\context projects\\ML\\codes and files\\project3.py" "project3.ipynb"')


# In this exercise, we want to design a category with the help of our knowledge of univariate normal distribution
# 
# Use the scikitlearn library to implement the models
# 
# https://scikit-learn.org/stable/

# # Read data
# 
# First, we read the data from the library and then divide the data into two categories, training and testing

# In[3]:


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

import numpy as np
from math import pi


# In[4]:


X[0]


# ### q1) print the number of train and test data and number of classes

# In[6]:


print("the number of train data is : {}".format(len(X_train)) )
print("the number of test data is {}: ".format(len(X_test)) )
print("there is {} different classes in the dataset".format(len(set(y))) )


# # Training a model assuming Gaussian distribution of data
# 
# Suppose we know that our data follows a Gaussian distribution and also that the features are independent of each other. Please teach a model for classifying classes with the help of the assumptions of the problem and without the help of ready-made models. Be careful that you should not use ready-made codes and you are only allowed to use ready-made functions for basic formulas such as average or variance.

# In[7]:


x_c0 = np.array([ X_train[i] for i in range( len(y_train) ) if y_train[i] == 0 ])
x_c1 = np.array([ X_train[i] for i in range( len(y_train) ) if y_train[i] == 1 ])
x_c2 = np.array([ X_train[i] for i in range( len(y_train) ) if y_train[i] == 2 ])


# In[8]:


#calculate mean and variance.
def get_mean_var(df):
    features_count = df.shape[1]
    mean , var = np.zeros(features_count) , np.zeros(features_count)
    for i in range(features_count):
        mean[i] = df.transpose()[i].mean()
        var[i]  = df.transpose()[i].var()
    return mean, var


# In[9]:


mean_c0, var_c0 = get_mean_var(x_c0)
mean_c1, var_c1 = get_mean_var(x_c1)
mean_c2, var_c2 = get_mean_var(x_c2)


# In[10]:


#calculate P(Ci)
p_c0 = len(x_c0) / len(y_train)
p_c1 = len(x_c1) / len(y_train)
p_c2 = len(x_c2) / len(y_train)


# In[11]:


def calc_p(X,mean,var):
    return (-np.log(np.sqrt(var))-((X-mean)**2)/(2*var))


# In[12]:


p = np.zeros((4,len(X_test)))

for i in range(len(X_test)):
    temp_0 =  calc_p( X_test[i,0], mean_c0[0], var_c0[0] )             + calc_p( X_test[i,1], mean_c0[1], var_c0[1] )             + calc_p( X_test[i,2], mean_c0[2], var_c0[2] )             + calc_p( X_test[i,3], mean_c0[3], var_c0[3] )
    
    G_0 = -.5*np.log(2*pi) + temp_0 + np.log(p_c0)

    temp_1 =  calc_p( X_test[i,0], mean_c1[0], var_c1[0] )             + calc_p( X_test[i,1], mean_c1[1], var_c1[1] )             + calc_p( X_test[i,2], mean_c1[2], var_c1[2] )             + calc_p( X_test[i,3], mean_c1[3], var_c1[3] )
        
    G_1 = -.5*np.log(2*pi) + temp_1 + np.log(p_c1)

    temp_2 =  calc_p( X_test[i,0], mean_c2[0], var_c2[0] )             + calc_p( X_test[i,1], mean_c2[1], var_c2[1] )             + calc_p( X_test[i,2], mean_c2[2], var_c2[2] )             + calc_p( X_test[i,3], mean_c2[3], var_c2[3] )
    
    G_2 = -.5*np.log(2*pi) + temp_2 + np.log(p_c2)
    
    g_0 = np.exp(G_0)
    g_1 = np.exp(G_1)
    g_2 = np.exp(G_2)
    
    p[0][i] = g_0/(g_1+g_0+g_2)
    p[1][i] = g_1/(g_1+g_0+g_2)
    p[2][i] = g_2/(g_1+g_0+g_2)
    
    p[3][i] = np.argmax([p[0][i], p[1][i], p[2][i]])


# Now measure the accuracy of your model with the help of test data

# In[13]:


y_pred_GNB = np.array(1 *(p[3]==y_test))
print('Accuracy =',(np.count_nonzero(y_pred_GNB)/len(y_pred_GNB)))


# # Training the model without knowing the data distribution
# 
# Classify the data with the help of SVM classifier as well as a simple neural network and compare the accuracy with the previous section

# In[15]:


from sklearn import svm
from sklearn import metrics

#train svm model
#write yor code here :

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))


# In[16]:


from sklearn.neural_network import MLPClassifier
#use two hidden layers 
#train multi layer perceptron
#write yor code here :

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20,20), max_iter=5000)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)

print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))


# # Read new data

# In[17]:


wine = datasets.load_wine()
# store the feature matrix (X) and response vector (y) 
X = wine.data 
y = wine.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 


# # Training a model assuming Gaussian distribution of data
# 
# Suppose we know that our data follows a Gaussian distribution. With the help of ready-made libraries, train a model for classifying classes. You can also use ready-made libraries for this part, and there is no need to implement.

# In[18]:


from sklearn.naive_bayes import GaussianNB
# training the model on training set 

# write your code here :

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# Now measure the accuracy of your model with the help of test data

# In[19]:


#write your code here :
print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))


# # Training the model without knowing the data distribution
# 
# Classify the data with the help of SVM classifier as well as a simple neural network and compare the accuracy with the previous section

# In[20]:


#train svm model

#write yor code here :

from sklearn.svm import SVC
from sklearn import metrics

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))


# In[21]:


#train multi layer perceptron

#write yor code here

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20,20), max_iter=5000)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)

print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))


# # Question

# In[22]:


import numpy as np


# In[23]:


def estimator(a,b,degree,offset = 0) :
    c = np.zeros([degree,degree])
    for i in range(degree):
        for j in range(degree):
            g = i + j
            c[i][j] = sum([(i-offset)**g for i in a])
    c2 = np.linalg.inv(c)
    d = np.zeros(degree)
    for i in range(degree) :
        d[i] = sum( [ b[j] * (a[j]-offset) ** i for j in range(len(a)) ] )
    return c,d, np.matmul(c2,d)


# In[24]:


X = [ 1394, 1395, 1396,1397, 1398 ]
Y = [12 ,19 ,29, 37 ,45]


# In[25]:


A,y,w = estimator(X,Y,3)


# In[26]:


print("A : \n",A)
print("y : \n",y)
print("w : \n",w)


# In[27]:


p1 = np.polyval(np.flip(w, 0), 1399)
print("Estimation for 1399 : {}".format(round(p1,2)))


# In[ ]:




