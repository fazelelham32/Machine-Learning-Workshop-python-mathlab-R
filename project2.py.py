#!/usr/bin/env python
# coding: utf-8

# In this exercise, we want to design a category with the help of our knowledge of univariate normal distribution
# 
# Use the scikitlearn library to implement the models
# 
# https://scikit-learn.org/stable/

# # Read data
# 
# First, we read the data from the library and then divide the data into two categories, training and testing

# In[1]:


import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

def print_accuracy(y_prd, y_true, name=''):
    acc = accuracy_score(y_prd, y_true, normalize=True) * 100
    print(name + 'Accuracy = {0}%'.format(round(acc, 2)))
    
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

# In[2]:


print("the number of train data is : ", len(y_train) )
print("the number of test data is : ", len(y_test) )
print("there is " + str(len(np.unique(y_train))) + " different classes in the dataset")


# # Training a model assuming Gaussian distribution of data
# 
# Suppose we know that our data follows a Gaussian distribution and also that the features are independent of each other. Please teach a model for classifying classes with the help of the assumptions of the problem and without the help of ready-made models. Be careful that you should not use ready-made codes and you are only allowed to use ready-made functions for basic formulas such as average or variance.

# In[3]:


# training the model on training set 

# write your code here :
train0, train1, train2 = X_train[y_train==0], X_train[y_train==1], X_train[y_train==2]
mu = np.vstack((train0.mean(axis=0), train1.mean(axis=0), train2.mean(axis=0)))
sigma = np.sqrt(np.vstack((train0.var(axis=0), train1.var(axis=0), train2.var(axis=0))))
P_C = len(train0) / len(y_train), len(train1) / len(y_train), len(train2) / len(y_train)


# اکنون با کمک داده های آزمایشی دقت مدل خود را بسنجید

# In[4]:


#write your code here :
def naive_bayes_classifier(X, mu, sigma, pc):
    N, K = X.shape[0], mu.shape[0]
    y = np.zeros((N, K))
    for c in range(K):
        y[:, c] = -np.asarray([sum(np.power(np.true_divide(np.subtract(x, mu[c]), sigma[c]), 2)) for x in X],
                              dtype='float32')/2 + np.log(pc[c])
    return np.argmax(y, axis=1)

print_accuracy(naive_bayes_classifier(X_train, mu, sigma, P_C), y_train, 'training data ')
print_accuracy(naive_bayes_classifier(X_test, mu, sigma, P_C), y_test, 'test data ')


# # Training the model without knowing the data distribution
# 
# Classify the data with the help of SVM classifier as well as a simple neural network and compare the accuracy with the previous section

# In[5]:


from sklearn import svm
#train svm model
#write yor code here :

for degree in (4, 9):
    clf = svm.SVC(kernel='poly', degree=degree)
    clf.fit(X_train, y_train)
    y = clf.predict(X_train)
    print('kernel: Polynomial with degree={0}'.format(degree))
    print_accuracy(clf.predict(X_train), y_train, 'training data ')
    print_accuracy(clf.predict(X_test), y_test, 'test data ')
    print()


# In[6]:


from sklearn.neural_network import MLPClassifier
#use two hidden layers 
#train multi layer perceptron
#write yor code here :
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
clf.fit(X_train, y_train)
y = clf.predict(X_train)
print_accuracy(clf.predict(X_train), y_train, 'training data ')
print_accuracy(clf.predict(X_test), y_test, 'test data ')


# # Read new data

# In[7]:


wine = datasets.load_wine()
# store the feature matrix (X) and response vector (y) 
X = wine.data 
y = wine.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 


# # Training a model assuming Gaussian distribution of data
# 
# Suppose we know that our data follows a Gaussian distribution. With the help of ready-made libraries, train a model for classifying classes. You can also use ready-made libraries for this part, and there is no need to implement.

# In[8]:


from sklearn.naive_bayes import GaussianNB
# training the model on training set 

# write your code here :
clf = GaussianNB()
clf.fit(X_train, y_train)


# Now measure the accuracy of your model with the help of test data

# In[9]:


#write your code here :
print_accuracy(clf.predict(X_train), y_train, 'training data ')
print_accuracy(clf.predict(X_test), y_test, 'test data ')


# # Training the model without knowing the data distribution
# 
# Classify the data with the help of SVM classifier as well as a simple neural network and compare the accuracy with the previous section

# In[10]:


#train svm model
#write yor code here :

from sklearn import svm
clf = svm.SVC(kernel='rbf', C=2.0)
clf.fit(X_train, y_train)
y = clf.predict(X_train)
print('kernel: RBF')
print_accuracy(clf.predict(X_train), y_train, 'training data ')
print_accuracy(clf.predict(X_test), y_test, 'test data ')


# In[11]:


#train multi layer perceptron
#write yor code here
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
clf.fit(X_train, y_train)
y = clf.predict(X_train)
print_accuracy(clf.predict(X_train), y_train, 'training data ')
print_accuracy(clf.predict(X_test), y_test, 'test data ')


# In[ ]:




