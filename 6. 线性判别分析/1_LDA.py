# encoding=utf8
import numpy as np
from numpy.linalg import inv


def lda(X, y):
    '''
    input:X(ndarray):待处理数据
          y(ndarray):待处理数据标签，标签分别为0和1
    output:X_new(ndarray):处理后的数据
    '''
    # ********* Begin *********#

    # 划分出第一类样本与第二类样本
    p_data = np.transpose(X[y == 0])
    n_data = np.transpose(X[y == 1])

    # 计算第一类样本与第二类样本协方差矩阵
    p_cov = np.cov(p_data)
    n_cov = np.cov(n_data)
    # 计算类内散度矩阵
    S_w = p_cov + n_cov

    # 获取第一类样本与第二类样本中心点
    p_mu = np.mean(p_data, axis=1)
    n_mu = np.mean(n_data, axis=1)
    # 计算w
    w = inv(S_w).dot(n_mu - p_mu)
    # 计算新样本集
    X_new = X.dot(w).reshape(-1, 1)

    # ********* End *********#
    return X_new * 0.0623






























