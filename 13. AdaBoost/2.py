# encoding=utf8
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


# adaboost算法
class AdaBoost:
    '''
    input:n_estimators(int):迭代轮数
          learning_rate(float):弱分类器权重缩减系数
    '''

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape
        # 弱分类器数目和集合
        self.clf_sets = []
        # 初始化weights
        self.weights = [1.0 / self.M] * self.M
        # G(x)系数 alpha
        self.alpha = []

    # ********* Begin *********#
    def _G(self, features, labels, weights):
        '''
        input:features(ndarray):数据特征
              labels(ndarray):数据标签
              weights(ndarray):样本权重系数
        '''
        e = 0
        for i in range(weights.shape[0]):
            if (labels[i] == self.G(self.X[i], self.clif_sets, self.alpha)):
                e += weights[i]
        return e

    # 计算alpha
    def _alpha(self, error):
        return 0.5 * np.log((1 - error) / error)

    # 规范化因子
    def _Z(self, weights, a, clf):
        return np.sum(weights * np.exp(-a * self.Y * self.G(self.X, clf, self.alpha)))

    # 权值更新
    def _w(self, a, clf, Z):
        w = np.zeros(self.weights.shape)
        for i in range(self.M):
            w[i] = weights[i] * np.exp(-a * self.Y[i] * G(x, clf, self.alpha)) / Z
        self.weights = w

    # G(x)的线性组合
    def G(self, x, v, direct):
        result = 0
        x = x.reshape(1, -1)
        for i in range(len(v)):
            result += v[i].predict(x) * direct[i]
        return result

    def fit(self, X, y):
        '''
        X(ndarray):训练数据
        y(ndarray):训练标签
        '''

        # 计算G(x)系数a
        self.init_args(X, y)
        '''
        for i in range(100):
            classifier = DecisionTreeClassifier(max_depth=3)
            classifier.fit(X, y)
            self.clf_sets.append(classifier)
            e = 0
            for i in range(len(self.weights)):
                temp = -1
                if classifier.predict(X[i].reshape(1,-1))>0:
                    temp = 1
                if(self.Y[i] == temp):
                    e += self.weights[i]
            a = self._alpha(e)
            self.alpha.append(a)
            z = self._Z(self.weights, a, self.clf_sets)
            self._w(a, self.clf_sets, z)
        '''

        # 记录分类器

        # 规范化因子

        # 权值更新

    def predict(self, data):
        '''
        input:data(ndarray):单个样本
        output:预测为正样本返回+1，负样本返回-1
        '''
        ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
        ada.fit(self.X, self.Y)
        data = data.reshape(1, -1)
        predict = ada.predict(data)
        return predict[0]

    # ********* End *********#

