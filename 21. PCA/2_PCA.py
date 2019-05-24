import numpy as np


def pca(data, k):
    """
    对data进行PCA，并将结果返回
    :param data:数据集，类型为ndarray
    :param k:想要降成几维，类型为int
    :return: 降维后的数据，类型为ndarray
    """

    # ********* Begin *********#
    # 零均值化
    mean = np.mean(data, axis=0)
    after_demean = data - mean

    cov = np.cov(after_demean.T)

    value, vector = np.linalg.eig(cov)

    sorted_vector = np.array([vec for val, vec in sorted(zip(value, vector), key=lambda x: x[0], reverse=True)])
    w = sorted_vector[:k]

    return np.dot(data, w.T)

    # ********* End *********#
