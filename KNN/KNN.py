#KNN模型的的实现

import numpy as np
from math import sqrt
from collections import Counter

def KNN_classify(k, x_train, y_train, x):

    #assert:make sure the data is satified
    assert 1 <= k <= x_train.shape[0], "k must be valid"
    assert x_train.shape[0] == y_train.shape[0], "the two size must be the same"
    assert x_train.shape[1] == x.shape[0],    "the two feature must be equal"

    distances = [sqrt(np.sum(X_train-x)**2) for X_train in x_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]

