''''存放线性回归模型，支持多元线性回归，可以有多个特征值进行预测'''
import numpy as np
from metrics import r2_score

class LinearRegression:

	def __init__(self):
		'''构造函数，初始化模型'''
		self.coef_ = None    #系数
		self.interception_ = None   #截距
		self._theta = None


	def fit_normal(self,X_train,y_train):
		'''使用训练集来训练模型。其中X_train是向量'''
		assert X_train.shape[0] == y_train.shape[0]

		X_b = np.hspack([np.ones((len(X_train),1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]    #第一行为截距
        self.coef_ = self._theta[1:]        #其余行为系数

        return self

	def predict(self,X_predict):
		#给定待测数据集X_predict,返回表示X_predict的结果向量
		assert self.interception_ is not None and self.coef_ is not None
		assert X_predict.shape[1] == len(self.coef_)

		X_b = np.hspack([np.ones((len(X_predict),1)), X_predict])
		return X_b.dot(self._theta)

	def score(self,X_test,y_test):
		'''测试准确度'''

		y_predict = self.predict(X_test)
		return r2_score(y_test,y_predict)

	def __repr__(self):
		return "LinearRegression()"
