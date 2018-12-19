import numpy as np


class PCA:

    def __init__(self, n_components):
        """构造函数，初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components   #用户传来的主成分维度 
        self.components_ = None   #不是用户传来的数据，而是根据用户传来的数据计算的结果，用户可以查询，所以加上_表示       

#根据scikit封装原则，有两个函数fit和transfer，fit根据用户传来的数据集计算出前k个主成分；transform根据fit的计算结果对数据进行降维操作
    def fit(self, X, eta=0.01, n_iters=1e4):
        """获得数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        def demean(X):
            '''对数据均值化，把所有样本均值归0'''
            return X - np.mean(X, axis=0)

        def f(w, X):
            '''求目标函数'''
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            '''求目标函数的梯度'''
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            '''方向向量的单位化'''
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            '''梯度上升法，求出参数'''
            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))  #初始化主成分
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i,:] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """将给定的X数据集，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        '''打印出相应的结果'''
        return "PCA(n_components=%d)" % self.n_components

