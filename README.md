#  ML-workshops

#  Machine-Learning:
Supervised Learning: 
Regression, Python and NumPy, Logistic Regression, Regularization, Neural Networks, Support Vector Machines
Unsupervided Learning: 
Clustering, Dimensionality Reduction and PCA, Anomally Detection, Recommender Systems

## PROJECT1:
**Line-by-Line Breakdown**

1. **`from sklearn import datasets`:** This imports the `datasets` module from the scikit-learn library. This module provides access to various pre-loaded datasets for machine learning, including the famous Iris dataset.

2. **`iris = datasets.load_iris()`:**  This line loads the Iris dataset and stores it in the variable `iris`. The Iris dataset is a classic dataset used in machine learning. It contains measurements of sepal and petal length and width for 150 samples of three different species of Iris flowers.

3. **`X = iris.data`:** The `iris.data` attribute holds the feature matrix, which is a 2D array where each row represents a sample and each column represents a feature (e.g., sepal length, sepal width, etc.). This line assigns this feature matrix to the variable `X`.

4. **`y = iris.target`:** The `iris.target` attribute contains the target labels, which indicate the species of Iris flower for each sample. This line assigns this target vector to the variable `y`.

5. **`print("our dataset has " + str(X.shape[1]) + " features. for more information about data surf the web")`:** This line calculates the number of features in the dataset using `X.shape[1]` and prints it along with a message about further information.

6. **`from sklearn.model_selection import train_test_split`:** This imports the `train_test_split` function from the `model_selection` module of scikit-learn. This function is crucial for splitting your data into training and testing sets.

7. **`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)`:**
   - This line splits the data into training and testing sets.
   - **`X, y`:** The feature matrix and target vector, respectively, are passed to the function.
   - **`test_size=0.4`:**  Specifies that 40% of the data should be allocated to the testing set.
   - **`random_state=1`:** Ensures that the split is deterministic, meaning that you'll get the same split every time you run the code with the same `random_state`. This is useful for reproducibility.
   - The function returns four arrays:
     - `X_train`: Features for the training set
     - `X_test`: Features for the testing set
     - `y_train`: Target labels for the training set
     - `y_test`: Target labels for the testing set

8. **`print("the number of train data is : " + str(X_train.shape[0]))`:** This line prints the number of samples (rows) in the training set.

9. **`print("the number of test data is : " + str(X_test.shape[0]))`:**  This line prints the number of samples in the testing set.

10. **`print("there are " + str(len(set(y_train))) + " different classes in the dataset")`:** This line determines the number of unique classes (species of Iris) present in the training data using `len(set(y_train))` and then prints it.


**Important Notes:**

- **Dataset Preparation:** The Iris dataset is relatively simple and already pre-processed. However, in real-world scenarios, you'll often need to perform additional data cleaning, feature engineering, and preprocessing before applying machine learning models.

- **Training and Testing:** The split between training and testing data is crucial in machine learning. Training data is used to build the model, while testing data is used to evaluate how well the model generalizes to unseen data. 

- **Univariate Normal Distribution:** The code snippet you provided doesn't explicitly involve working with univariate normal distribution. However, the concept of normal distribution is important in statistical analysis and can be used in various aspects of machine learning, including feature analysis, model assumptions, and hypothesis testing. 


### Project1-1
Let's break down the code and understand how it implements a Bayesian classifier assuming Gaussian distributions.

**Code Breakdown**

1. **`class Bayes():`** 
   - Defines a class called `Bayes` to encapsulate the Bayesian classification logic.

2. **`__init__(self):`**
   - This is the constructor of the `Bayes` class. 
   - `self.priors = None`: Initializes the `priors` attribute, which will store the prior probabilities of each class. It's set to `None` initially.
   - `self.distribution_params = None`: Initializes the `distribution_params` attribute, which will store the parameters (mean and variance) of the Gaussian distributions for each feature and each class. It's also set to `None` initially.

3. **`gaussian(self, x, u, var):`**
   - This method implements the Gaussian probability density function (PDF). It calculates the probability density at a given point `x` for a Gaussian distribution with mean `u` and variance `var`. 
   - **Formula:** $$P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{ -\frac{(x-\mu)^2}{2\sigma^2}}$$
   - Where:
      - `x`: The data point
      - `u`: The mean of the Gaussian distribution
      - `var`: The variance of the Gaussian distribution

4. **`_compute_priors(self, yt):`**
   - This method calculates the prior probabilities for each class.
   - It first finds the unique classes (`classes`) from the target labels (`yt`).
   - Then, it calculates the number of samples belonging to each class and divides by the total number of samples to get the prior probabilities.

5. **`_estimate_pdf_dist(self, xt, yt):`**
   - This method estimates the parameters of the Gaussian distribution (mean and variance) for each feature and each class.
   - It iterates through each class and each feature, calculating the mean (`np.mean`) and variance (`np.var`) of the feature values for samples belonging to that class.

6. **`train(self, xt, yt):`**
   - This method trains the Bayesian model.
   - It calls `_compute_priors` to calculate the prior probabilities based on the training labels (`yt`).
   - It calls `_estimate_pdf_dist` to estimate the Gaussian distribution parameters for each feature and class based on the training features (`xt`) and labels (`yt`).

7. **`predict(self, sample):`**
   - This method predicts the class of a given sample.
   - It calculates the posterior probability for each class by multiplying the prior probability of the class with the likelihood of the sample belonging to that class. 
   - The likelihood is calculated as the product of the Gaussian probabilities of each feature in the sample, given the estimated parameters for that feature and class.
   - Finally, it returns the class with the highest posterior probability.

8. **`model = Bayes()`:**
   - Creates an instance of the `Bayes` class, effectively creating a Bayesian model.

9. **`model.train(X_train, y_train):`**
   - Trains the model on the training data (`X_train` and `y_train`).

**Testing the Model's Accuracy**

After training, you would evaluate the model's accuracy using the test data (`X_test` and `y_test`). You would do this by:

1. **Predicting:** Use the trained `model.predict(X_test)` to predict the class labels for the test data.
2. **Comparing:** Compare the predicted labels with the actual labels (`y_test`) to calculate metrics such as accuracy, precision, recall, or F1-score.

**Key Points**

- **Naive Bayes:** This code implements a simplified version of Naive Bayes. The "naive" assumption is that features are independent of each other. This is a common assumption in Naive Bayes models.
- **Gaussian Assumption:** The code specifically assumes that the data for each feature follows a Gaussian distribution within each class. 
- **No Libraries:** The code avoids using pre-built classification models from libraries like scikit-learn. It's designed to demonstrate the fundamental logic of a Bayesian classifier. 


**Code Block 1**

```python
#write your code here:
predictions = []
for I in range(X_test.shape[0]):
 predictions.append(model.predict(X_test[I,:]))

from sklearn.metrics import precision_recall_fscore_support
prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions, average = 'weighted')
print(“Precision:”+str(round(prec*100,2))+” Recall:”+str(round(rcall*100,2))+" F1-score:"+str(round(f1*100 ,2)))
```

**Explanation:**

1. **`predictions = []`:**  Initializes an empty list named `predictions`. This list will store the model's predictions on the test data.
2. **`for I in range(X_test.shape[0]):`:** Starts a `for` loop that iterates over each row of the `X_test` dataset. `X_test.shape[0]` gives the number of rows in the `X_test` dataset.
3. **`predictions.append(model.predict(X_test[I,:]))`:** Inside the loop, it makes a prediction using the trained `model` for the current row (`X_test[I,:]`) and appends the predicted value to the `predictions` list.
4. **`from sklearn.metrics import precision_recall_fscore_support`:**  Imports the `precision_recall_fscore_support` function from the `sklearn.metrics` module. This function is used to calculate evaluation metrics.
5. **`prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions, average = 'weighted')`:** Calculates the precision, recall, F1-score, and support for the model's predictions. 
    - `y_test` is the true labels for the test data.
    - `predictions` is the list of predictions made by the model.
    - `average='weighted'` calculates the weighted average of the metrics across all classes, which is useful when you have an imbalanced dataset.
    - `_` is a placeholder to discard the support value.
6. **`print(“Precision:”+str(round(prec*100,2))+” Recall:”+str(round(rcall*100,2))+" F1-score:"+str(round(f1*100 ,2)))`:** Prints the calculated precision, recall, and F1-score as percentages, rounded to two decimal places.

**Code Block 2**

```python
from sklearn import svm
#train svm model
#write your code here:
svm_model = svm.SVC()
svm_model.fit(X_train,y_train)
# predict
predictions = svm_model.predict(X_test)

# evaluate performance
prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions, average = 'weighted')
print("Precision:"+str(round(prec*100,2))+"% Recall:"+str(round(rcall*100,2))+"% F1-score:"+str(round(f1 *100,2))+"%")
```

**Explanation:**

1. **`from sklearn import svm`:** Imports the `svm` module from `sklearn`, which contains the support vector machine (SVM) algorithms.
2. **`svm_model = svm.SVC()`:** Creates an instance of the `SVC` (Support Vector Classification) class. This is a basic SVM classifier.
3. **`svm_model.fit(X_train,y_train)`:**  Trains the SVM model using the training data `X_train` and corresponding labels `y_train`.
4. **`predictions = svm_model.predict(X_test)`:** Uses the trained SVM model (`svm_model`) to predict labels for the test data `X_test`. 
5. **`prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions, average = 'weighted')`:**  Calculates precision, recall, F1-score, and support for the SVM model's predictions using the `precision_recall_fscore_support` function (explained in the first code block). 
6. **`print("Precision:"+str(round(prec*100,2))+"% Recall:"+str(round(rcall*100,2))+"% F1-score:"+str(round(f1 *100,2))+"%")`:** Prints the precision, recall, and F1-score as percentages, rounded to two decimal places. 

**Code Block 3**

```python
from sklearn.neural_network import MLPClassifier
#use two hidden layers
#train multi layer perceptron
#write your code here:
neural_model = MLPClassifier(hidden_layer_sizes=(256,100 ), activation='logistic',max_iter=1000)
neural_model.fit(X_train,y_train)

# predict
predictions = neural_model.predict(X_test)

# evaluate
prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions, average = 'weighted')
print("Precision:"+str(round(prec*100,2))+"% Recall:"+str(round(rcall*100,2))+"% F1-score:"+str(round(f1 *100,2))+"%")
```

**Explanation:**

1. **`from sklearn.neural_network import MLPClassifier`:** Imports the `MLPClassifier` class from `sklearn.neural_network`. This class is used for training multi-layer perceptron (MLP) neural networks.
2. **`neural_model = MLPClassifier(hidden_layer_sizes=(256,100 ), activation='logistic',max_iter=1000)`:** Creates an instance of the `MLPClassifier` with:
    - `hidden_layer_sizes=(256,100)`: Defines two hidden layers in the neural network, one with 256 neurons and the other with 100 neurons.
    - `activation='logistic'`: Sets the activation function to be the logistic sigmoid function.
    - `max_iter=1000`: Sets the maximum number of iterations for the training process to 1000.
3. **`neural_model.fit(X_train,y_train)`:** Trains the neural network model (`neural_model`) using the training data `X_train` and labels `y_train`.
4. **`predictions = neural_model.predict(X_test)`:** Uses the trained neural network model (`neural_model`) to predict labels for the test data `X_test`.
5. **`prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions, average = 'weighted')`:** Calculates the precision, recall, F1-score, and support for the neural network model's predictions, as done in the previous blocks.
6. **`print("Precision:"+str(round(prec*100,2))+"% Recall:"+str(round(rcall*100,2))+"% F1-score:"+str(round(f1 *100,2))+"%")`:** Prints the precision, recall, and F1-score as percentages, rounded to two decimal places.

**In Summary:**

This code demonstrates how to train and evaluate two different machine learning models (SVM and a neural network) on a dataset. It calculates important classification metrics and prints them for comparison. 

Let's break down the code snippets and understand the operations.

**Code Snippet 1: Loading and Splitting the Wine Dataset**

```python
# Read new data
wine = datasets.load_wine()
# store the feature matrix (X) and response vector (y)
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
```

* **Line 1:** `wine = datasets.load_wine()`:  This line loads the "wine" dataset using the `load_wine()` function from the `datasets` module of scikit-learn. This dataset is a classic machine learning benchmark dataset.
* **Line 2:** `X = wine.data`:  This assigns the feature matrix (attributes describing the wines) of the dataset to the variable `X`. This will be used to train and test models.
* **Line 3:** `y = wine.target`: This assigns the response vector (class labels - types of wine) to the variable `y`. This is what we want to predict.
* **Line 4:** `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)`: This line uses the `train_test_split` function from scikit-learn to divide the dataset into training and testing sets.  
    * `test_size=0.4` means 40% of the data will be used for testing.
    * `random_state=1` ensures the split is consistent across different runs.

**Code Snippet 2: Training a Gaussian Naive Bayes Model**

```python
from sklearn.naive_bayes import GaussianNB
# training the model on the training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)
```

* **Line 1:** `from sklearn.naive_bayes import GaussianNB`:  This line imports the `GaussianNB` class from scikit-learn's `naive_bayes` module. The Gaussian Naive Bayes algorithm assumes that features are normally (Gaussian) distributed.
* **Line 2:** `gnb = GaussianNB()`: Creates an instance of the `GaussianNB` class and assigns it to the variable `gnb`.
* **Line 3:** `gnb.fit(X_train, y_train)`:  This line trains the Gaussian Naive Bayes model. The `fit` method takes the training data (`X_train`) and the corresponding labels (`y_train`) and calculates the parameters necessary for the model to make predictions.

**Code Snippet 3: Evaluating the Gaussian Naive Bayes Model**

```python
# write your code here:
predictions = gnb.predict(X_test)

prec, rcall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
print("Precision:" + str(round(prec * 100, 2)) + "% Recall:" + str(round(rcall * 100, 2)) + "% F1-score:" + str(round(f1 * 100, 2)) + "%")
```

* **Line 1:** `predictions = gnb.predict(X_test)`: Uses the trained `gnb` model to make predictions on the test data (`X_test`) and stores the results in the `predictions` variable.
* **Line 2:** `prec, rcall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')`:  This line calculates several performance metrics:
    * **precision:**  The proportion of correctly predicted positive cases out of all cases predicted as positive.
    * **recall:** The proportion of correctly predicted positive cases out of all actual positive cases.
    * **f1-score:**  A harmonic mean of precision and recall, providing a balanced measure of the model's performance.
    * **support:** The number of actual occurrences of each class in the test set.
    * **average='weighted'**: This tells the function to compute the weighted average of the metrics across all classes, considering the class support.
* **Line 3:**  `print("Precision:" + str(round(prec * 100, 2)) + "% Recall:" + str(round(rcall * 100, 2)) + "% F1-score:" + str(round(f1 * 100, 2)) + "%")`: This line prints the calculated precision, recall, and F1-score as percentages, rounded to two decimal places.

**Code Snippet 4: Training and Evaluating a Support Vector Machine (SVM) Model**

```python
from sklearn import svm
#train svm model
svm_model = svm.SVC()
svm_model.fit(X_train,y_train)

# predict
predictions = svm_model.predict(X_test)

# evaluate
prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions, average = 'weighted')
print("Precision:"+str(round(prec*100,2))+"% Recall:"+str(round(rcall*100,2))+"% F1-score:"+str(round(f1 *100,2))+"%")
```

* **Line 1:** `from sklearn import svm`: Imports the `svm` module from scikit-learn, which contains the Support Vector Machine algorithms.
* **Line 2:** `svm_model = svm.SVC()`: Creates an instance of the `SVC` (Support Vector Classifier) class and assigns it to the variable `svm_model`.
* **Line 3:** `svm_model.fit(X_train, y_train)`:  Trains the SVM model using the training data.
* **Line 4:** `predictions = svm_model.predict(X_test)`: Uses the trained SVM model to make predictions on the test data.
* **Lines 5-6:** This part is identical to the evaluation section in the Gaussian Naive Bayes code, calculating and printing the precision, recall, and F1-score for the SVM model's performance.

Let me know if you have any more code snippets you'd like me to explain or if you'd like a deeper dive into any of these concepts.

# project2

## Code Snippet 1:  Training a Multi-Layer Perceptron (MLP)

```python
from sklearn.neural_network import MLPClassifier
#train multi layer perceptron
#write yor code here :
neural_model = MLPClassifier(hidden_layer_sizes=(256,100 ), activation='logistic',max_iter=1000)
neural_model.fit(X_train,y_train)
```

1. **Line 1:** `from sklearn.neural_network import MLPClassifier`: This imports the `MLPClassifier` class from scikit-learn's `neural_network` module.  The `MLPClassifier` implements a feedforward artificial neural network.
2. **Line 3:** `neural_model = MLPClassifier(hidden_layer_sizes=(256,100 ), activation='logistic',max_iter=1000)`: Creates an instance of the `MLPClassifier` class and assigns it to the variable `neural_model`. The key parameters are:
    * `hidden_layer_sizes=(256,100)`:  Defines the architecture of the neural network.  It has two hidden layers, with 256 neurons in the first layer and 100 neurons in the second layer.
    * `activation='logistic'`: Specifies the activation function used in the neurons. 'logistic' refers to the sigmoid function, which outputs a value between 0 and 1.
    * `max_iter=1000`: Sets the maximum number of iterations (epochs) for the training process.
3. **Line 4:** `neural_model.fit(X_train,y_train)`: Trains the MLP model using the training data (`X_train`) and labels (`y_train`). The `fit` method adjusts the weights and biases of the neural network to learn the patterns in the data.

## Code Snippet 2:  Making Predictions with the Trained MLP

```python
# predict
predictions = neural_model.predict(X_test)
```

1. **Line 1:** `predictions = neural_model.predict(X_test)`: Uses the trained `neural_model` to make predictions on the test data (`X_test`). The `predict` method uses the learned weights and biases to classify the input features in `X_test`.

## Code Snippet 3: Evaluating the MLP Model

```python
# evaluate
prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions,average = 'weighted')
print("Precision:"+str(round(prec*100,2))+"%  Recall:"+str(round(rcall*100,2))+"%  F1-score:"+str(round(f1*100,2))+"%")
```

1. **Line 1:** `prec,rcall,f1,_ = precision_recall_fscore_support(y_test, predictions,average = 'weighted')`:  Calculates several performance metrics:
    * **precision:** The proportion of correctly predicted positive cases out of all cases predicted as positive.
    * **recall:** The proportion of correctly predicted positive cases out of all actual positive cases.
    * **f1-score:** A harmonic mean of precision and recall, providing a balanced measure of the model's performance.
    * **support:** The number of actual occurrences of each class in the test set.
    * **average='weighted'**:  Computes the weighted average of the metrics across all classes, taking into account the class support.
2. **Line 2:**  `print("Precision:"+str(round(prec*100,2


## Code Explanations:

**Training a model assuming Gaussian distribution of data**

```python
from sklearn.naive_bayes import GaussianNB
# training the model on the training set
clf = GaussianNB()
clf.fit(X_train, y_train)
```

1. **`from sklearn.naive_bayes import GaussianNB`**: This line imports the `GaussianNB` class from the `sklearn.naive_bayes` module. This class implements the Gaussian Naive Bayes algorithm, which is specifically designed for data that follows a Gaussian distribution.
2. **`clf = GaussianNB()`**: This line creates an instance of the `GaussianNB` class and assigns it to the variable `clf`. This object represents the Gaussian Naive Bayes model.
3. **`clf.fit(X_train, y_train)`**: This line trains the Gaussian Naive Bayes model using the training data. `X_train` represents the features of the training data, and `y_train` represents the corresponding labels. The `fit()` method calculates the parameters of the Gaussian distribution for each feature based on the training data.

**Measuring the accuracy of the model**

```python
print_accuracy(clf.predict(X_train), y_train, 'training data')
print_accuracy(clf.predict(X_test), y_test, 'test data')
```

1. **`print_accuracy(clf.predict(X_train), y_train, 'training data')`**: This line calculates the accuracy of the model on the training data. It first uses the trained `clf` model to predict the labels of the training data using `clf.predict(X_train)`. Then, it calls the `print_accuracy` function, passing the predicted labels, the actual training labels (`y_train`), and the string "training data" as arguments. The `print_accuracy` function likely calculates and prints the accuracy score. 
2. **`print_accuracy(clf.predict(X_test), y_test, 'test data')`**:  This line does the same as the previous line but for the test data. It predicts the labels of the test data using `clf.predict(X_test)`, then calls `print_accuracy` with the predictions, actual test labels (`y_test`), and the string "test data".

**Training the model without knowing the data distribution**

**SVM classifier:**

```python
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=2.0)
clf.fit(X_train, y_train)
y = clf.predict(X_train)
print('kernel: RBF')
print_accuracy(clf.predict(X_train), y_train, 'training data')
print_accuracy(clf.predict(X_test), y_test, 'test data')
```

1. **`from sklearn import svm`**: This line imports the `svm` module from the `sklearn` library. This module contains classes for implementing Support Vector Machines (SVMs).
2. **`clf = svm.SVC(kernel='rbf', C=2.0)`**: This line creates an instance of the `SVC` class, which is a specific type of SVM classifier. It sets the kernel to `'rbf'` (Radial Basis Function) and the regularization parameter `C` to `2.0`.
3. **`clf.fit(X_train, y_train)`**: This line trains the SVM classifier using the training data. 
4. **`y = clf.predict(X_train)`**: This line predicts the labels of the training data using the trained SVM model and stores them in the variable `y`.
5. **`print('kernel: RBF')`**: This line prints the string "kernel: RBF" to the console, indicating the type of kernel used for the SVM. 
6. **`print_accuracy(clf.predict(X_train), y_train, 'training data')`**: This line calculates and prints the accuracy of the SVM model on the training data, similar to the previous example.
7. **`print_accuracy(clf.predict(X_test), y_test, 'test data')`**: This line calculates and prints the accuracy of the SVM model on the test data.

**Neural Network:**

```python
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
clf.fit(X_train, y_train)
y = clf.predict(X_train)
print_accuracy(clf.predict(X_train), y_train, 'training data')
print_accuracy(clf.predict(X_test), y_test, 'test data')
```

1. **`from sklearn.neural_network import MLPClassifier`**: This line imports the `MLPClassifier` class from the `sklearn.neural_network` module. This class represents a multi-layer perceptron (MLP), a type of artificial neural network.
2. **`clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)`**: This line creates an instance of the `MLPClassifier` class. It specifies the architecture of the neural network with two hidden layers, each containing 100 neurons. It also sets the maximum number of iterations for training to 1000.
3. **`clf.fit(X_train, y_train)`**: This line trains the neural network using the training data. 
4. **`y = clf.predict(X_train)`**: This line predicts the labels of the training data using the trained neural network.
5. **`print_accuracy(clf.predict(X_train), y_train, 'training data')`**: This line calculates and prints the accuracy of the neural network on the training data.
6. **`print_accuracy(clf.predict(X_test), y_test, 'test data')`**: This line calculates and prints the accuracy of the neural network on the test data.

**Bonus Section:**

Yes, there are ways to infer the distribution of a dataset before choosing a classifier. Some common methods include:

* **Visualizing the data:** Histograms, scatter plots, and other visualizations can help identify potential distributions.
* **Statistical tests:** Tests like the Kolmogorov-Smirnov test or the Shapiro-Wilk test can assess whether the data follows a specific distribution.
* **Density estimation:** Techniques like kernel density estimation can estimate the probability density function of the data.

This information can then inform the selection of a suitable classifier. For example:

* If the data appears to follow a Gaussian distribution, a Gaussian Naive Bayes classifier might be a good choice.
* If the data is not normally distributed, other classifiers like SVM or neural networks might be more suitable.

**Markdown and LaTeX:**

The LaTeX code `\alpha^2` represents the square of alpha (`α`). 
The Markdown for this would be: `$\alpha^2$`.

The `$$` symbols are used to wrap the LaTeX code. 



## PROJECT3
**Code Breakdown:**

**1.  Calculating Mean and Variance**

```python
#calculate mean and variance.
def get_mean_var(df):
  features_count = df.shape[1]
  mean , var = np.zeros(features_count) , np.zeros(features_count)
  for i in range(features_count):
    mean[i] = df.transpose()[i].mean()
    var[i] = df.transpose()[i].var()
  return mean, var

mean_c0, var_c0 = get_mean_var(x_c0)
mean_c1, var_c1 = get_mean_var(x_c1)
mean_c2, var_c2 = get_mean_var(x_c2)
```

* **`def get_mean_var(df):`**: This defines a function named `get_mean_var` that takes a DataFrame (`df`) as input. 
* **`features_count = df.shape[1]`**: This line calculates the number of columns in the input DataFrame (`df`) and stores it in the `features_count` variable.  The `shape` attribute of a DataFrame gives you the number of rows and columns as a tuple.  `shape[1]` specifically accesses the second element of this tuple, which is the number of columns. 
* **`mean , var = np.zeros(features_count) , np.zeros(features_count)`**:  This line initializes two NumPy arrays, `mean` and `var`, each of length `features_count` and filled with zeros. These arrays will store the calculated means and variances for each feature (column) of the DataFrame.
* **`for i in range(features_count):`**: This loop iterates through each feature (column) in the DataFrame.
* **`mean[i] = df.transpose()[i].mean()`**:  This line calculates the mean of the `i`-th column of the transposed DataFrame.  
    * `df.transpose()` transposes the DataFrame, switching rows and columns.
    * `[i]` selects the `i`-th column of the transposed DataFrame.
    * `.mean()` calculates the mean of the selected column.
* **`var[i] = df.transpose()[i].var()`**: Similar to the line above, this line calculates the variance of the `i`-th column of the transposed DataFrame.
* **`return mean, var`**:  The function returns the calculated means and variances as a tuple.

* **`mean_c0, var_c0 = get_mean_var(x_c0)`**:  This line calls the `get_mean_var` function with the DataFrame `x_c0` as input, and the returned mean and variance values are stored in the variables `mean_c0` and `var_c0`, respectively.  The code repeats this for data frames `x_c1` and `x_c2`.

**2.  Calculating Probabilities of Classes**

```python
#calculate P(Ci)
p_c0 = len(x_c0) / len(y_train)
p_c1 = len(x_c1) / len(y_train)
p_c2 = len(x_c2) / len(y_train)
```

* **`p_c0 = len(x_c0) / len(y_train)`**: This calculates the probability of class "C0" by dividing the number of samples in `x_c0` by the total number of samples in the training set (`y_train`). The code repeats this for classes "C1" and "C2" using `x_c1` and `x_c2`, respectively.

**3.  Calculating the Probability Density Function (PDF) for a Feature**

```python
def calc_p(X, mean, var):
  return (-np.log(np.sqrt(var))-((X-mean)**2)/(2*var))
```

* **`def calc_p(X, mean, var):`**:  This defines a function named `calc_p` that takes three arguments:
    * `X`: A single data point for a specific feature.
    * `mean`: The mean of the feature (for a specific class).
    * `var`: The variance of the feature (for a specific class).
* **`return (-np.log(np.sqrt(var))-((X-mean)**2)/(2*var))`**: This line calculates the probability density of the data point `X` given the mean and variance of the feature.  The formula used is the PDF for a normal distribution. This function helps calculate the likelihood of a given data point belonging to a specific class, based on the class's mean and variance for that feature.

**4.  Applying the Gaussian Naive Bayes Model**

```python
p = np.zeros((4,len(X_test)))

for i in range(len(X_test)):
  temp_0 = calc_p( X_test[i,0], mean_c0[0], var_c0[0] ) \
  + calc_p( X_test[i,1], mean_c0[1], var_c0[1] ) \
  + calc_p( X_test[i,2], mean_c0[2], var_c0[2] ) \
  + calc_p( X_test[i,3], mean_c0[3], var_c0[3] )

  G_0 = -.5*np.log(2*pi) + temp_0 + np

Let's break down the code snippets you've provided, focusing on what they do and their purpose within the context of machine learning.

**1.  Training with SVM (Support Vector Machines)**

```python
from sklearn import svm
from sklearn import metrics

#train svm model
clf = svm.SVC(kernel='linear')  # Create an SVM model with a linear kernel
clf.fit(X_train, y_train)       # Train the model on the training data
y_pred = clf.predict(X_test)    # Make predictions on the test data
print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) )) # Calculate and print accuracy
```

**Explanation:**

* **`from sklearn import svm`**:  Imports the `svm` module from scikit-learn, which provides various support vector machine algorithms.
* **`from sklearn import metrics`**: Imports the `metrics` module, which provides tools for evaluating model performance (like accuracy).
* **`clf = svm.SVC(kernel='linear')`**: Creates an SVM classifier (`clf`). 
    * `kernel='linear'` specifies that a linear kernel will be used for the SVM.  This means the model will try to find a separating hyperplane that is linear.
* **`clf.fit(X_train, y_train)`**:  Trains the SVM model using the training data `X_train` (features) and `y_train` (corresponding labels). This step is where the model learns the patterns in the data.
* **`y_pred = clf.predict(X_test)`**: Uses the trained SVM model to predict the labels for the test data `X_test`.  
* **`print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))`**: Calculates and prints the accuracy of the SVM model.
    * `metrics.accuracy_score(y_test, y_pred)` calculates the accuracy by comparing the predicted labels (`y_pred`) with the actual labels (`y_test`).
    * `round(..., 2)` rounds the accuracy to two decimal places.
    * `print("Accuracy: {}%".format(...))` formats the output to display the accuracy as a percentage.

**2.  Training with Multi-Layer Perceptron (MLP)**

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20,20), max_iter=5000)  # Create MLP with 2 hidden layers
mlp.fit(X_train,y_train)                                       # Train the MLP
y_pred = mlp.predict(X_test)                                     # Make predictions
print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) )) # Calculate and print accuracy
```

**Explanation:**

* **`from sklearn.neural_network import MLPClassifier`**: Imports the `MLPClassifier` class from the `neural_network` module. This class implements a multi-layer perceptron, a type of artificial neural network.
* **`mlp = MLPClassifier(hidden_layer_sizes=(20,20), max_iter=5000)`**: Creates an MLP classifier (`mlp`).
    * `hidden_layer_sizes=(20, 20)` specifies the architecture of the network: two hidden layers, each with 20 neurons.
    * `max_iter=5000` sets the maximum number of iterations the training algorithm will run for.
* **`mlp.fit(X_train,y_train)`**: Trains the MLP model using the training data.  This step involves adjusting the weights and biases of the network to learn the relationships in the data.
* **`y_pred = mlp.predict(X_test)`**: Uses the trained MLP model to make predictions on the test data.
* **`print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))`**: Calculates and prints the accuracy of the MLP model. This step is the same as in the SVM example.

**3.  Reading New Data (Wine Dataset)**

```python
wine = datasets.load_wine()
# store the feature matrix (X) and response vector (y)
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
```

**Explanation:**

* **`wine = datasets.load_wine()`**: Loads the "wine" dataset from scikit-learn's `datasets` module. This dataset contains information about different types of wines and their chemical properties.
* **`X = wine.data`**:  Extracts the feature matrix from the loaded dataset and stores it in the variable `X`. The feature matrix contains the chemical measurements for each wine sample.
* **`y = wine.target`**:  Extracts the target labels (wine types) from the dataset and stores them in `y`.
* **`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)`**:  Splits the data into training and test sets using `train_test_split`.
    * `X, y`: The data to split.
    * `test_size=0.4`:  Specifies that 40% of the data will be used for testing, and the remaining 60% will be used for training.
    * `random_state=1`: Sets a random seed for reproducibility.  This ensures that the same data samples will be assigned to training and test sets each time you run the code.

**Training with Gaussian Assumption**

You're correct! Since you're assuming a Gaussian distribution for your data, you can use a Gaussian Naive Bayes classifier.  Here's how you might do that:

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create a Gaussian Naive Bayes model
model = GaussianNB()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

**Key Points**

* **Gaussian Naive Bayes:** Assumes that features are independently distributed according to a Gaussian (normal) distribution.
* **Efficiency:**  Naive Bayes models are generally very fast to train and make predictions.
* **Pros and Cons:**  They can perform well when the Gaussian assumption holds. However, if the data significantly deviates from Gaussianity, performance might suffer.

**Important Considerations**

* **Data Exploration:** Before you use any classifier, it's always a good idea to explore your data visually (histograms, scatter plots) to see if the Gaussian assumption is reasonable.
* **Feature Engineering:** Sometimes, transforming your features can help them better conform to a Gaussian distribution or improve model performance.

Let me know if you have any more questions or would like to explore more advanced techniques for handling data with unknown distributions! 


You're absolutely right!  Let's break down each code snippet line by line, focusing on the concepts and how they fit together:

**1. Gaussian Naive Bayes Classifier**

```python
from sklearn.naive_bayes import GaussianNB
# training the model on the training set

# write your code here:

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

1. **`from sklearn.naive_bayes import GaussianNB`**:  This line imports the `GaussianNB` class from the scikit-learn library's `naive_bayes` module. The `GaussianNB` class implements a Gaussian Naive Bayes classifier.
2. **`clf = GaussianNB()`**:  Creates an instance of the `GaussianNB` classifier and assigns it to the variable `clf`. This creates a new, untrained Gaussian Naive Bayes model.
3. **`clf.fit(X_train, y_train)`**: Trains the classifier using the training data (`X_train`) and corresponding target labels (`y_train`).  The `fit()` method learns the parameters of the Gaussian distributions for each feature and class, allowing the model to make predictions on unseen data.
4. **`y_pred = clf.predict(X_test)`**:  Uses the trained classifier (`clf`) to predict the class labels for the test data (`X_test`). The `predict()` method applies the learned parameters to the test data to make predictions.

**2. Calculating Accuracy**

```python
#write your code here:
print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))
```

1. **`print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))`**: Calculates and prints the accuracy of the classifier's predictions. Let's break this down further:
   * **`metrics.accuracy_score(y_test, y_pred)`**: Uses the `accuracy_score` function from the `metrics` module to calculate the accuracy of the predictions (`y_pred`) against the true labels (`y_test`).  Accuracy is the proportion of correct predictions.
   * **`round(..., 2)`**: Rounds the accuracy score to two decimal places for better readability.
   * **`format(..., "{}%")`**:  Formats the output as a string with the rounded accuracy score followed by a percentage sign. 
   * **`print(...)`**: Displays the formatted accuracy score.

**3. SVM Classifier**

```python
#train svm model

#write your code here:

from sklearn.svm import SVC
from sklearn import metrics

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))
```

1. **`from sklearn.svm import SVC`**:  Imports the `SVC` class from the `sklearn.svm` module. The `SVC` class implements Support Vector Machines for classification.
2. **`from sklearn import metrics`**: Imports the `metrics` module for calculating model evaluation metrics like accuracy.
3. **`clf = SVC(kernel='linear')`**:  Creates an instance of the `SVC` classifier using a linear kernel.  A linear kernel assumes that the data can be separated by a straight line in the feature space.  
4. **`clf.fit(X_train, y_train)`**:  Trains the SVM classifier using the training data. 
5. **`y_pred = clf.predict(X_test)`**:  Uses the trained SVM classifier to make predictions on the test data.
6. **`print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))`**: Calculates and prints the accuracy of the SVM model, using the same logic as before.

**4. Multi-layer Perceptron (MLP)**

```python
#train multi layer perceptron

#write your code here

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20,20), max_iter=5000)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)

print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))
```

1. **`from sklearn.neural_network import MLPClassifier`**: Imports the `MLPClassifier` class from the `sklearn.neural_network` module. The `MLPClassifier` implements a multi-layer perceptron, a type of artificial neural network.
2. **`mlp = MLPClassifier(hidden_layer_sizes=(20,20), max_iter=5000)`**:  Creates an instance of the `MLPClassifier`.
   * `hidden_layer_sizes=(20, 20)`:  Specifies that the network has two hidden layers, each with 20 neurons.
   * `max_iter=5000`:  Sets the maximum number of iterations the training algorithm will run for.
3. **`mlp.fit(X_train, y_train)`**: Trains the MLP classifier using the training data.  This involves adjusting the weights and biases of the network to learn the relationships in the data.
4. **`y_pred = mlp.predict(X_test)`**: Uses the trained MLP model to make predictions on the test data.
5. **`print("Accuracy: {}%".format( round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 ) ))`**: Calculates and prints the accuracy of the MLP model, using the same logic as before.

**5. Polynomial Regression Estimator**

```python
import numpy as np
def estimator(a,b,degree,offset = 0):
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

X = [ 1394, 1395, 1396, 1397, 1398 ]
Y = [12, 19, 29, 37, 45]

A,y,w = estimator


## PROJECT4
The code snippets you provided are using Python and the Pandas library to read CSV files containing data for a machine learning project. Here's a breakdown of what the code does and the expected outcomes:

**Project Goal:**

This project appears to be about classifying data points into different categories. The data has two characteristics (features) and belongs to one of three classes. The goal is to find the best model for this classification task. 

**The Code:**

* **`import pandas as pd`:** This line imports the Pandas library, a powerful tool for data manipulation and analysis in Python.
* **`pd.read_csv(r"C:\Users\dor_mah\Music\importent folder\main folder\context projects\ML\codes and files\project4-files\train_50.csv")`:** This line reads a CSV file named "train_50.csv" located at the specified path. This file likely contains the training data with 50 data points.
* **`pd.read_csv(r"C:\Users\dor_mah\Music\importent folder\main folder\context projects\ML\codes and files\project4-files\test_labels.csv")`:** This line reads a CSV file named "test_labels.csv". This file likely contains the true labels or classes for the test data.

**What the code does:**

1. **Data Loading:**  The code loads training data from CSV files. You have three different sets: "train_50.csv", "train_250.csv", and "train_800.csv", suggesting you'll be comparing models trained on different amounts of data.
2. **Evaluation:** The code instructs you to evaluate the performance of four different methods on the test data. This likely refers to Table 5.1, which is not included in your provided code. This table would contain descriptions of the four classification methods you need to compare.
3. **Model Selection:**  After evaluating the models, you will need to determine which one performs best.
4. **Best Model Application:**  The final section instructs you to apply the best performing model to a new dataset ("test_without_labels.csv") and save the predicted labels to a file. 

**Expected Outcomes:**

* You will have trained four different classification models on each of the three training datasets.
* You will have evaluated the performance of these models on the "test_labels.csv" dataset.
* You will have identified the best performing model.
* You will have used this best model to predict the classes of data points in "test_without_labels.csv".

**Important Note:**

You'll need to add code to:

* **Read the test labels from "test_labels.csv"**
* **Implement the four classification methods mentioned in Table 5.1**
* **Calculate performance metrics (like accuracy) for each model**
* **Choose the best performing model**
* **Apply the best model to predict the classes in "test_without_labels.csv"**
* **Save the predicted labels to a file**

Let me know if you have more details about Table 5.1 or the classification methods you are working with, and I can help you with the implementation. 




# PROJECT5
There is a one-dimensional dataset, blue class data (C1) follows Gaussian distribution with σ = 0 and μ = 0.1, and red σ class (C2) follows Gaussian distribution with σ = 0.25 and μ = 0.1. The number of data in each class σ is the same.
Obtain the Bayes decision boundary equation. (Gusi is defined as follows)
G(x)= 1√2πσ2e−(x−μ)22σ2
Answer:
The decision equation is obtained by combining the distributions of each class. By equating the two distributions above
And solving the equation, the following value is obtained:
x = 0.125
It is placed in the attachment of the aforementioned documents. The first column is the attribute and the second column is its class.
Draw the two-category distribution function and the equation of the decision boundary and the data (data.csv) in a graph.
Show the distribution function and data of each class with different colors.
Answer:
Using the equation that you obtained, the accuracy of Bayes on the test data
Get test.csv.
Answer:
Full code in:
https://colab.research.google.com/drive/1ZvTrfa0MXy56xoc5w5SpVf7uLwPTl-og?usp=drive_open









# Machine_Learning in Python: 
machine learning using python language to implement different algorithms
Using existing packages for machine learning,  Implementing our machine Learning algorithms and models

Prerequisites:

    A basic knowledge of Python programming language.
    A good understaning of Machine Learning.
    Linear Algebra
