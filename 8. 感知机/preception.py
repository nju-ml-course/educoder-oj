# encoding=utf8
import numpy as np


# 构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        # 编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.] * data.shape[1])
        self.b = np.array([1.])
        for i in range(self.max_iter):
            for row in range(data.shape[0]):
                if label[row] * (np.dot(data[row], np.transpose(self.w)) + self.b) < 0:
                    self.w += self.lr * label[row] * data[row]
                    self.b += self.lr * label[row]

    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        z = np.dot(data, np.transpose(self.w)) + self.b
        return [1 if item > 0 else -1 for item in z]

