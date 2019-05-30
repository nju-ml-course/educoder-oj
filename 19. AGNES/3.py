#encoding=utf8
from sklearn.cluster import AgglomerativeClustering

def Agglomerative_cluster(data):
    '''
    对红酒数据进行聚类
    :param data: 数据集，类型为ndarray
    :return: 聚类结果，类型为ndarray
    '''

    #********* Begin *********#
    mean = data.mean()         #计算平均数
    deviation = data.std()     #计算标准差
    # 标准化数据的公式: (数据值 - 平均数) / 标准差
    data = (data - mean) / deviation
    agnes = AgglomerativeClustering(n_clusters=3)
    result = agnes.fit_predict(data)
    return result

    #********* End *********#
