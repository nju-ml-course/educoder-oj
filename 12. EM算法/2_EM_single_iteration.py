import numpy as np
from scipy import stats
from collections import Counter


def em_single(init_values, observations):
    """
    模拟抛掷硬币实验并估计在一次迭代中，硬币A与硬币B正面朝上的概率
    :param init_values:硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.2, 0.7]代表硬币A正面朝上的概率为0.2，硬币B正面朝上的概率为0.7。
    :param observations:抛掷硬币的实验结果记录，类型为list。
    :return:将估计出来的硬币A和硬币B正面朝上的概率组成list返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
    """

    # ********* Begin *********#
    def get_likehood(p, l):
        likehood = 1
        for i in l:
            if i == 1:
                likehood *= p
            else:
                likehood *= 1 - p
        return likehood

    exist_matrix = np.zeros((2, 2))
    p_a, p_b = init_values[0], init_values[1]
    for experiment in observations:
        likehood_a = get_likehood(p_a, experiment)
        likehood_b = get_likehood(p_b, experiment)
        prob_a = likehood_a / (likehood_a + likehood_b)
        prob_b = likehood_b / (likehood_a + likehood_b)
        c = Counter(experiment)
        exist_matrix[0][0] += prob_a * c[1]
        exist_matrix[0][1] += prob_a * c[0]
        exist_matrix[1][0] += prob_b * c[1]
        exist_matrix[1][1] += prob_b * c[0]
    new_p_a = exist_matrix[0][0] / (exist_matrix[0][0] + exist_matrix[0][1])
    new_p_b = exist_matrix[1][0] / (exist_matrix[1][0] + exist_matrix[1][1])
    return [new_p_a, new_p_b]
    # ********* End *********#
