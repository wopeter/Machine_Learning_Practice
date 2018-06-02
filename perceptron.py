# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:13:24 2018

@author: pengte
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

np.random.seed(123)



#Dataset
X, y = make_blobs(n_samples=100, centers=2)
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Dataset")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

y_true = y[:, None]

X_train, X_test, y_train, y_test = train_test_split(X, y_true)
print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {X_test.shape}')

class Perceptron():
    def __init__(self):
        pass
    
    def train(self, X, y, learning_rate=0.05, n_iters=100):
        n_samples, n_features = X.shape
        
        #Step 0: Initialize the parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        for i in range(n_iters):
            #step 1: Compute the activation
            a = np.dot(X, self.weights) + self.bias
            
            #step 2: Compute the output
            y_predict = self.step_function(a)
            
            #step 3: Compute weight updates
            delta_w = learning_rate * np.dot(X.T, (y - y_predict))
            delta_b = learning_rate * np.sum(y - y_predict)
            
            #step 4: Update the parameters
            self.weights+= delta_w
            self.bias += delta_b
            
        return self.weights, self.bias
    
    def step_function(self, x):
        return np.array([1 if elem >= 0 else 0 for elem in x])[:, None]
    
    def predict(self, X):
        a = np.dot(X, self.weights) + self.bias
        return self.step_function(a)
    
def plot_hyperplane(X, y, weights, bias):
    """
    Plots the dataset and the estimated decision hyperplane
    """
    slope = - weights[0] / weights[1] #斜率
    intercept = - bias / weights[1] #截距
    x_hyperplane = np.linspace(-10, 10, 10)
    y_hyperplane = slope * x_hyperplane + intercept
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(x_hyperplane, y_hyperplane, "-")
    plt.title("Dataset and fitted decision hyperplane")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()
if __name__ == "__main__":
    
    #Initialization and training the model
    p = Perceptron()
    w_trained, b_trained = p.train(X_train, y_train, learning_rate=0.05, n_iters=500)
    
    #Testing
    y_p_train = p.predict(X_train)
    y_p_test = p.predict(X_test)
    
    print(f"training accuracy: {100 - np.mean(np.abs(y_p_train - y_train)) * 100}%")
    print(f"training accuracy: {100 - np.mean(np.abs(y_p_test - y_test)) * 100}%")
    
    #Visualize decision boundary
    
    plot_hyperplane(X, y, w_trained, b_trained)
    
    
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
