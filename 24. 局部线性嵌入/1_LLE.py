# encoding=utf8
import numpy as np


def find_neighbors(data, i, k):
    dist = sorted(range(len(data)), key=lambda x: np.linalg.norm(data[x] - data[i]))
    return set(dist[1: k + 1])


def cal_c_jk(data, i, j, k):
    return np.dot((data[i] - data[j]), (data[i] - data[k]))


def lle(data, d, k):
    """
    input:data(ndarray):待降维数据,行数为样本个数，列数为特征数
          d(int):降维后数据维数
          k(int):最近的k个样本
    output:Z(ndarray):降维后的数据
    """
    # ********* Begin *********#
    m = len(data)
    W = np.zeros((m, m))
    for i in range(m):
        # 确定样本i的邻域
        neighbors = find_neighbors(data, i, k)
        lower = sum(1 / cal_c_jk(data, i, l, s) for l in neighbors for s in neighbors)
        for j in neighbors:
            # 求矩阵c及其逆
            upper = sum(1 / cal_c_jk(data, i, j, k) for k in neighbors)
            # 求w
            W[i][j] = upper / lower

    # 求得M并矩阵分解
    I = np.identity(m)
    M = np.dot((I - W).T, (I - W))

    value, vector = np.linalg.eig(M)
    index = np.argsort(value)[: d]
    # 求Z(z1; z2; z3; ...; zm) 每一行为一个新的降维投影
    Z = vector[:, index].T
    # ********* End *********#
    return Z

