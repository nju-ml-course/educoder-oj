import numpy as np


def avg(feature, pred, c):
    feature_c = feature[pred == c]
    m = len(feature_c)
    mu = np.mean(feature_c, axis=0)
    return 1 / m * sum(np.linalg.norm(fea - mu) for fea in feature_c)


def d_cen(feature, pred, c1, c2):
    feature_c1 = feature[pred == c1]
    feature_c2 = feature[pred == c2]
    mu1 = np.mean(feature_c1, axis=0)
    mu2 = np.mean(feature_c2, axis=0)
    return np.linalg.norm(mu1 - mu2)


def d_min(feature, pred, c1, c2):
    feature_c1 = feature[pred == c1]
    feature_c2 = feature[pred == c2]
    return min(np.linalg.norm(f1 - f2) for f1 in feature_c1 for f2 in feature_c2)


def diam(feature, pred, c):
    feature_c = feature[pred == c]
    m = len(feature_c)
    if m == 1:
        return 0
    return max(np.linalg.norm(feature_c[i] - feature_c[j]) for i in range(m) for j in range(i + 1, m))


def calc_DBI(feature, pred):
    """
    计算并返回DB指数
    :param feature: 待聚类数据的特征，类型为`ndarray`
    :param pred: 聚类后数据所对应的簇，类型为`ndarray`
    :return: DB指数
    """

    # ********* Begin *********#
    class_set = set(pred)
    return 1 / len(class_set) * sum(
        max(
            (avg(feature, pred, i) + avg(feature, pred, j)) / d_cen(feature, pred, i, j)
            for j in class_set if j != i)
        for i in class_set)
    # ********* End *********#


def calc_DI(feature, pred):
    """
    计算并返回Dunn指数
    :param feature: 待聚类数据的特征，类型为`ndarray`
    :param pred: 聚类后数据所对应的簇，类型为`ndarray`
    :return: Dunn指数
    """

    # ********* Begin *********#
    class_set = list(set(pred))
    m = len(class_set)
    lower = max(diam(feature, pred, c) for c in class_set)
    return min(d_min(feature, pred, class_set[i], class_set[j])
               for i in range(m) for j in range(i+1, m)) / lower
    # ********* End *********#


