# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:44:52 2018

@author: pengte
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
np.random.seed(123)

#Dataset
#We will use the digits dataset as an example. It consists of the 1797 images of hand-written digits.
#is represented by a 64-dimensional vector of pixel values.

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

#Example digits
fig = plt.figure(figsize=(10, 8))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    plt.imshow(X[i].reshape((8, 8)), cmap="gray")
    
plt.show()

class kNN():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.data = X
        self.targets = y
    
    #计算欧氏距离
    def eucliden_distance(self, X):
        #input: single data point
        #当X仅为一个输入时
        if X.ndim == 1:
            l2 = np.sqrt(np.sum((self.data - X) ** 2, axis=1))
        #当X为多个输入时， n_samples为样本的个数
        if X.ndim == 2:
            n_samples, _ = X.shape
            l2 = [np.sqrt(np.sum((self.data - X[i]) ** 2, axis=1)) for i in range(n_samples)]
        return np.array(l2)
    
    def predict(self, X, k=1):
        
        dists = self.eucliden_distance(X)
        
        if X.ndim == 1:
            if k == 1:
                nn = np.argmin(dists)
                return self.targets[nn]
            else:
                knn = np.argsort(dists)[:k]
                y_knn = self.targets[knn]
                max_vote = max(y_knn, key=list(y_knn).count)
                return max_vote
            
        if X.ndim == 2:
            knn = np.argsort(dists)[:, :k]
            y_knn = self.targets[knn]
            if k == 1:
                return y_knn.T
            else:
                n_samples, _ = X.shape
                max_votes = [max(y_knn[i], key=list(y_knn[i]).count) for i in range(n_samples)]
                return max_votes
            
if __name__ == "__main__":
    
    #Initializiong and training the model
    knn = kNN()
    knn.fit(X_train, y_train)
    print("Testing one datapoint, k=1")
    print(f"Predicted label: {knn.predict(X_test[0], k=1)}")
    print(f"True label: {y_test[0]}")
    print()
    print("Testing one datapoint, k=5")
    print(f"Predicted label: {knn.predict(X_test[20], k=5)}")
    print(f"True label: {y_test[20]}")
    print()
    print("Testing 10 datapoint, k=1")
    print(f"Predicted labels:{knn.predict(X_test[5:15], k=1)}")
    print(f"True labels: {y_test[5:15]}")
    print()
    print("Testing 10 datapoint, k=4")
    print(f"Predicted labels: {knn.predict(X_test[5:15], k=4)}")
    print(f"True labels: {y_test[5:15]}")
    print()
    
    #Accuracy on test set
    y_p_test1 = knn.predict(X_test, k=1)
    test_acc1 = np.sum(y_p_test1[0] == y_test) / len(y_p_test1[0]) * 100
    print(f"Test accuracy with k = 1: {format(test_acc1)}")
    
    y_p_test5 = knn.predict(X_test, k=5)
    test_acc5 = np.sum(y_p_test5 == y_test) / len(y_p_test5) * 100
    print(f"Test accuracy with k = 5: {format(test_acc5)}")
    
    
    
    