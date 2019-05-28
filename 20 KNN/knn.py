# encoding=utf8
import numpy as np


class kNNClassifier(object):
    def __init__(self, k):
        '''
        初始化函数
        :param k:kNN算法中的k
        '''
        self.k = k
        # 用来存放训练数据，类型为ndarray
        self.train_feature = None
        # 用来存放训练标签，类型为ndarray
        self.train_label = None

    def fit(self, feature, label):
        """
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        """
        self.train_feature = feature
        self.train_label = label
        self.data = np.concatenate((feature, np.transpose([label])), axis=1)

    def predict(self, feature):
        """
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        """

        # ********* Begin *********#
        def computeDistance(X, Y):
            return np.linalg.norm(np.subtract(X, Y))

        def moMax(X):
            return np.argmax(np.bincount(X))

        ans = []
        for row in feature:
            arr = sorted(self.data, key=lambda item: computeDistance(item[:-1], row))[:self.k + 1]
            ans.append(moMax([row[-1] for row in arr]))
        return ans


