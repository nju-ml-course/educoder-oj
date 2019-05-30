import numpy as np


def calc_min_dist(cluster1, cluster2):
    '''
    计算簇间最小距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的最小距离
    '''

    #********* Begin *********#
    dis = 100000000
    for vec1 in cluster1:
        for vec2 in cluster2:
            dis=min(dis, np.linalg.norm(vec1-vec2))
    return dis

    #********* End *********#


def calc_max_dist(cluster1, cluster2):
    '''
    计算簇间最大距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的最大距离
    '''

    #********* Begin *********#
    dis = 0
    for vec1 in cluster1:
        for vec2 in cluster2:
            dis=max(dis, np.linalg.norm(vec1-vec2))
    return dis

    #********* End *********#


def calc_avg_dist(cluster1, cluster2):
    '''
    计算簇间平均距离
    :param cluster1:簇1中的样本数据，类型为ndarray
    :param cluster2:簇2中的样本数据，类型为ndarray
    :return:簇1与簇2之间的平均距离
    '''

    #********* Begin *********#
    dis = 0
    for vec1 in cluster1:
        for vec2 in cluster2:
            dis+=np.linalg.norm(vec1-vec2)
    return dis/(cluster1.shape[0]*cluster2.shape[0])

    #********* End *********#