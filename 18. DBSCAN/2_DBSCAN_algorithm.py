# encoding=utf8
import numpy as np
import random
from copy import copy
from collections import deque


# 寻找eps邻域内的点
def findNeighbor(j, X, eps):
    return {p for p in range(X.shape[0]) if np.linalg.norm(X[j] - X[p]) <= eps}


# dbscan算法
def dbscan(X, eps, min_Pts):
    """
    input:X(ndarray):样本数据
          eps(float):eps邻域半径
          min_Pts(int):eps邻域内最少点个数
    output:cluster(list):聚类结果
    """
    # ********* Begin *********#

    # 初始化核心对象集合
    core_objects = {i for i in range(len(X)) if len(findNeighbor(i, X, eps)) >= min_Pts}

    # 初始化聚类簇数
    k = 0

    # 初始化未访问的样本集合
    not_visited = set(range(len(X)))

    # 初始化聚类结果
    cluster = np.zeros(len(X))

    while len(core_objects) != 0:
        old_not_visited = copy(not_visited)
        # 初始化聚类簇队列
        o = random.choice(list(core_objects))
        queue = deque()
        queue.append(o)
        not_visited.remove(o)

        while len(queue) != 0:
            q = queue.popleft()
            neighbor_list = findNeighbor(q, X, eps)
            if len(neighbor_list) >= min_Pts:
                # 寻找在邻域中并没被访问过的点
                delta = neighbor_list & not_visited
                for element in delta:
                    queue.append(element)
                    not_visited.remove(element)

        k += 1
        this_class = old_not_visited - not_visited
        cluster[list(this_class)] = k
        core_objects = core_objects - this_class

    # ********* End *********#
    return cluster
