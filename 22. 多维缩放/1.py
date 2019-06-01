# -*- coding: utf-8 -*-
import numpy as np


def mds(data, d):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维数
    output:Z(ndarray):降维后的数据
    '''
    # ********* Begin *********#
    # 计算dist2,dist2i,dist2j,dist2ij

    # 计算B

    # 矩阵分解得到特征值与特征向量

    # 计算Z

    # ********* End *********#
    DSquare = np.zeros([data.shape[0], data.shape[0]])
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            DSquare[i][j] = np.sum(np.square(data[i] - data[j]))
    totalMean = np.mean(DSquare)
    rowMean = np.mean(DSquare, axis=1)
    columnMean = np.mean(DSquare, axis=0)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = np.linalg.eigh(B)  # 求特征值及特征向量
    # 对特征值进行排序，得到排序索引
    eigValSorted_indices = np.argsort(-eigVal)
    # 提取d个最大特征向量
    topd_eigVec = eigVec[:, eigValSorted_indices[:d]]
    X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[eigValSorted_indices[:d]])))
    return X




















