#模拟sklean，写的封装KNN函数

import numpy as np
from math import sqrt
from collections import Counter

class anKNNClassifier:

    def __init__(self,k):  #构造函数
        """初始化分类器"""
        assert k >= 1, "k must be valid"
        self.k = k;
        self._X_train = None
        self._y_train = None

    def fit(self,x_train,y_train):
        """根据训练集X_train和y_train训练KNN分类器"""

        self._X_train = x_train
        self._y_train = y_train
        return self

    def predict(self,X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1],     "the feature number must be equal"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1], "the feature must be equal"

        distances = [sqrt(np.sum(x_train - x)**2) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_commot(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k
