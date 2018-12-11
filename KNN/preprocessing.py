'''2.第二步进行归一化处理，仿造sklearn封装了归一化类'''

import numpy as np

class StandardScaler:

	def __init__(self):
		self.mean_ = None
		self.scale_ = None;

	'''根据训练集X来获得数据的均值和方差'''
	def fit(self,X):
		assert X.ndim == 2, "the dimension of X must be 2"

		self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])]) 
		self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

		return self

	'''将X根据StandardScaler进行均值方差归一化处理'''
	def tranform(self,X):
		assert X.ndim == 2, "the dimension must be 2"

		resX = np.empty(shape=X.shape,dtype=float)
		for col in range(X.shape[1]):
			resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]
		return resX