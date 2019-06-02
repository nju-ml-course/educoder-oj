# encoding=utf8
import numpy as np


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


def cal_dis(old_centroids, centroids):
    dis = 0
    for i in range(old_centroids.shape[0]):
        dis += np.linalg.norm(old_centroids[i] - centroids[i], 2)
    return dis


class Kmeans():
    """Kmeans聚类算法.
    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数.
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon,
        则说明算法已经收敛
    """

    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        np.random.seed(1)

    # ********* Begin *********#
    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        m, n = X.shape
        center = np.zeros((self.k, n))
        for i in range(self.k):
            index = int(np.random.uniform(0, m))
            center[i] = X[index]
        return center

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        return np.argsort(distances)[0]

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        m, n = X.shape
        clusters = np.mat(np.zeros((m, 1)))
        for i in range(m):
            index = self._closest_centroid(X[i], centroids)
            clusters[i] = index
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        centroids = np.zeros([self.k, X.shape[1]])
        for i in range(self.k):
            pointsInCluster = []
            for j in range(clusters.shape[0]):
                if clusters[j] == i:
                    pointsInCluster.append(X[j])
            centroids[i] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        return null

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)
        clusters = []
        iter = 0
        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        while iter < self.max_iterations:
            iter += 1

            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)

            # 计算新的聚类中心
            old_centroids = centroids[:]
            centroids = self.update_centroids(clusters, X)
            if cal_dis(old_centroids, centroids) < self.varepsilon:
                break

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
        return np.array(clusters).reshape([X.shape[0], ])

    # ********* End *********#

























