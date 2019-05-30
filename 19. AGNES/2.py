import numpy as np

def dist(cluster1, cluster2):
    dis = 100000000
    for vec1 in cluster1:
        for vec2 in cluster2:
            dis=min(dis, np.linalg.norm(vec1-vec2))
    return dis

def find_Min(M):
    m = 100000000
    x = 0
    y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if M[i][j] < m:
                x = i
                y = j
    return x, y, m

def AGNES(feature, k):
    '''
    AGNES聚类并返回聚类结果，量化距离时请使用簇间最大欧氏距离
    假设数据集为`[1, 2], [10, 11], [1, 3]]，那么聚类结果可能为`[[1, 2], [1, 3]], [[10, 11]]]
    :param feature:数据集，类型为ndarray
    :param k:表示想要将数据聚成`k`类，类型为`int`
    :return:聚类结果，类型为list
    '''

    #********* Begin *********#
    #初始化C和M
    C = [];M = []
    for i in feature:
        Ci = []
        Ci.append(i)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(dist(i, j))
        M.append(Mi)
    q = len(C)
    #合并更新
    while q > k:
        x, y, min = find_Min(M)
        C[x].extend(C[y])
        C.remove(C[y])
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist(i, j))
            M.append(Mi)
        q -= 1
    return C

    #********* End *********#

