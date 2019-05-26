# encoding=utf8
import numpy as np


# mse
def mse_score(y_predict, y_test):
    mse = np.mean((y_predict - y_test) ** 2)
    return mse


# r2
def r2_score(y_predict, y_test):
    '''
    input:y_predict(ndarray):预测值
          y_test(ndarray):真实值
    output:r2(float):r2值
    '''
    # ********* Begin *********#
    upper = sum((p - y) ** 2for p, y in zip(y_predict, y_test))
    lower = sum((y_test.mean() - y) ** 2 for y in y_test)
    r2 = 1 - upper / lower
    # ********* End *********#
    return r2


class LinearRegression:
    def __init__(self):
        """初始化线性回归模型"""
        self.theta = None

    def fit_normal(self, train_data, train_label):
        """
        input:train_data(ndarray):训练样本
              train_label(ndarray):训练标签
        """
        # ********* Begin *********#
        ones = np.ones((len(train_data), 1))
        train_data = np.column_stack((train_data, ones))
        self.theta = np.linalg.inv(train_data.T @ train_data) @ train_data.T @ train_label
        # ********* End *********#
        return self

    def predict(self, test_data):
        """
        input:test_data(ndarray):测试样本
        """
        # ********* Begin *********#
        ones = np.ones((len(test_data), 1))
        test_data = np.column_stack((test_data, ones))
        return test_data @ self.theta
        # ********* End *********#
