import numpy as np
from sklearn.metrics import accuracy_score


class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}  # 标记概率
        self.label_indexes = {}  # 不同类别标记,在数据集中对应的行
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def fit(self, feature, label):
        """
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        """

        def store_prop():
            m = len(feature)  # 获取行数
            n = len(feature[0])  # 获取列数
            for i, item in enumerate(label):
                if item not in self.label_indexes.keys():
                    self.label_indexes[item] = [i]
                else:
                    self.label_indexes[item].append(i)
            for labelItem in self.label_indexes.keys():
                # 拉普拉斯修正
                self.label_prob[labelItem] = (len(self.label_indexes[labelItem]) + 1) / (
                        m + len(self.label_indexes.keys()))
                # 不使用拉普拉斯修正
                # self.label_prob[labelItem] = len(self.label_indexes[labelItem]) / m
            # ------------------------------
            # store the condition prop
            for labelItem in self.label_indexes.keys():  # for every label
                self.condition_prob[labelItem] = {}
                # subRows = feature[self.label_indexes[labelItem]]  # 获取label对应的某些行
                subRows = [row for i, row in enumerate(feature)
                           if i in self.label_indexes[labelItem]]
                for i in range(n):  # for every column (x_i)
                    if i == 2:
                        tmpDic = {1: 0, 2: 0, 3: 0}
                    else:
                        tmpDic = {1: 0, 2: 0}

                    for row in subRows:
                        if row[i] not in tmpDic.keys():
                            tmpDic[row[i]] = 1
                        else:
                            tmpDic[row[i]] += 1
                    count = len(list(tmpDic.values()))
                    for k, v in tmpDic.items():
                        tmpDic[k] = (v + 1) / (len(subRows) + count)
                    self.condition_prob[labelItem][i] = tmpDic
        store_prop()
        return self

    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''

        result = []
        # 对每条测试数据都进行预测
        for i, f in enumerate(feature):
            # 可能的类别的概率
            prob = np.zeros(len(self.label_prob.keys()))
            ii = 0
            for label, label_prob in self.label_prob.items():
                # 计算概率
                prob[ii] = label_prob
                for j in range(len(feature[0])):
                    prob[ii] *= self.condition_prob[label][j][f[j]]
                ii += 1
            # 取概率最大的类别作为结果
            result.append(list(self.label_prob.keys())[np.argmax(prob)])
        return np.array(result)


# boston = db.load_iris()
# X = boston.data
# y = boston.target
X = [[1, 2, 3],
     [1, 1, 3],
     [2, 1, 3],
     [2, 2, 1],
     [2, 2, 2],
     [2, 1, 3],
     [1, 2, 3],
     [1, 2, 3],
     [1, 2, 3],
     [1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]
y = [1, 0, 1, 0, 1]
bayes = NaiveBayesClassifier()

bayes.fit(X, y)
predict = bayes.predict(X)
print(accuracy_score(y, predict))
