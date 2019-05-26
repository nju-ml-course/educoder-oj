import numpy as np
from sklearn.tree import DecisionTreeClassifier


class BaggingClassifier(object):
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        # 分类器的数量，默认为10
        self.n_model = n_model
        # 用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []

    def fit(self, feature, label):
        '''
        训练模型，请记得将模型保存至self.models
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        '''
        self.models = [DecisionTreeClassifier(max_depth=3).fit(feature, label) for _ in range(self.n_model)]

    def predict(self, feature):
        '''
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        tmp_arr = np.transpose([clf_.predict(feature) for clf_ in self.models])
        predict = []
        for row in tmp_arr:
            dic = {}
            for item in row:
                if item not in dic.keys():
                    dic[item] = 1
                else:
                    dic[item] += 1
            predict.append(list(max(dic.items(), key=lambda d: d[1]))[0])
        return predict


