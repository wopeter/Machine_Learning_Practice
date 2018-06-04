# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:55:19 2018

@author: pengte
"""

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    
