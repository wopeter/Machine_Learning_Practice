# -*- coding: utf-8 -*-
"""
Created on Wed May 30 21:28:32 2018

@author: pengte
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))

