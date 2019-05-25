import random

import numpy as np
# 建议代码，也算是Begin-End中的一部分
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier():
    def __init__(self, n_model=10):
        '''
        初始化函数
        '''
        # 分类器的数量，默认为10
        self.n_model = n_model
        # 用于保存模型的列表，训练好分类器后将对象append进去即可
        self.models = []
        # 用于保存决策树训练时随机选取的列的索引
        self.col_indexs = []
        self.feature_k = 3

    def fit(self, feature, label):
        """
        训练模型
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: None
        """

        def random_sampling(X, y):
            """
            自助采样
            :param X:
            :param y:
            :return: 自助采样之后的结果
            """
            m, n = np.shape(X)
            # 有放回抽取
            row_indexes = [random.randint(0, m - 1) for _ in range(m)]
            # 选取随机k个特征
            col_indexes = random.sample(range(n), self.feature_k)

            X_res = [[X[index][col] for col in col_indexes] for index in row_indexes]
            y_res = [y[index] for index in row_indexes]
            return X_res, y_res, col_indexes

        for i in range(self.n_model):
            X, y, cols = random_sampling(feature, label)
            self.col_indexs.append(cols)
            self.models.append(DecisionTreeClassifier(max_depth=4).fit(X, y))

    def predict(self, feature):
        '''
        :param feature:测试集数据，类型为ndarray
        :return:预测结果，类型为ndarray，如np.array([0, 1, 2, 2, 1, 0])
        '''
        # ************* Begin ************#
        tmp_arr = np.transpose(
            [clf.predict(np.array(feature[:, self.col_indexs[i]])) for i, clf in enumerate(self.models)])
        predict = []
        for row in tmp_arr:
            di = {}
            for item in row:
                if item not in di.keys():
                    di[item] = 1
                else:
                    di[item] += 1
            predict.append(list(max(di.items(), key=lambda d: d[1]))[0])
        return predict
        # ************* End **************#
