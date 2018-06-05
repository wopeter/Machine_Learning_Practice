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

def selectJrand(i, m):
	j = i
	while(j == i):
		j = int(random.uniform(0, m))
	return j
	
def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter): 
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	m, n = shape(dataMatrix)
	
	b = 0
	alphas = mat(zeros((m, 1)))
	
	iter = 0
	while(iter < maxIter):
		pass
		
	return b, alphas

def calcWs(alphas, dataArr, classLabels):
	X = mat(dataArr)
	labelMat = mat(classLabels).transpose()
	m, n = shape(X)
	w = zeros((n, 1))
	for i in range(m):
		w += multiply(alphas[i] * labelMat[i], X[i, :].T)
	return w
	
def plotfig_SVM(xMat, yMat, ws, b, alphas):
	xMat = mat(xMat)
	yMat = mat(yMat)
	
	b = array(b)[0]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])
	x = arange(-1.0, 10.0, 0.1)
	y = (-b - ws[0, 0]*x)/ws[1, 0]
	ax.plot(x, y)
	for i in range(shape(yMat[0, :])[1]):
		if yMat[0, i] > 0:
			ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
		else:
			ax.plot(xMat[i, 0], xMat[i, 1], 'kp')
	for i in range(100):
		if alphas[i] > 0.0
			ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
	plt.show()
	
if __name__ == "__main__":
	pass
	
    
