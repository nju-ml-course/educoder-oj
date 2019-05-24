from sklearn.metrics.cluster import fowlkes_mallows_score, adjusted_rand_score


def cluster_performance(y_true, y_pred):
    """
    返回Rand指数和FM指数
    :param y_true:参考模型的簇划分，类型为ndarray
    :param y_pred:聚类模型给出的簇划分，类型为ndarray
    :return: Rand指数，FM指数
    """
    # ********* Begin *********#
    rand = adjusted_rand_score(y_true, y_pred)
    fm = fowlkes_mallows_score(y_true, y_pred)
    return fm, rand
    # ********* End *********#
