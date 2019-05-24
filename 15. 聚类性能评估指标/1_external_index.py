import numpy as np


def count_pairs(y_true, y_pred):
    m = len(y_true)
    SS, SD, DS, DD = 0, 0, 0, 0
    for i in range(m):
        for j in range(i + 1, m):
            if y_pred[i] == y_pred[j] and y_true[i] == y_true[j]:
                SS += 1
            elif y_pred[i] == y_pred[j] and y_true[i] != y_true[j]:
                SD += 1
            elif y_pred[i] != y_pred[j] and y_true[i] == y_true[j]:
                DS += 1
            else:
                DD += 1
    return SS, SD, DS, DD


def calc_JC(y_true, y_pred):
    """
    计算并返回JC系数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: JC系数
    """

    # ******** Begin *******#
    a, b, c, d = count_pairs(y_true, y_pred)
    return a / (a + b + c)

    # ******** End *******#


def calc_FM(y_true, y_pred):
    """
    计算并返回FM指数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: FM指数
    """

    # ******** Begin *******#
    a, b, c, d = count_pairs(y_true, y_pred)
    return a / np.sqrt((a + b) * (a + c))
    # ******** End *******#


def calc_Rand(y_true, y_pred):
    """
    计算并返回Rand指数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: Rand指数
    """

    # ******** Begin *******#
    a, b, c, d = count_pairs(y_true, y_pred)
    m = len(y_true)
    return 2 * (a + d) / (m * (m - 1))
    # ******** End *******#
