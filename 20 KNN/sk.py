from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def classification(train_feature, train_label, test_feature):
    '''
    对test_feature进行红酒分类
    :param train_feature: 训练集数据，类型为ndarray
    :param train_label: 训练集标签，类型为ndarray
    :param test_feature: 测试集数据，类型为ndarray
    :return: 测试集数据的分类结果
    '''

    # 实例化StandardScaler对象
    scaler = StandardScaler()
    # 用data的均值和标准差来进行标准化，并将结果保存到after_scaler
    X = scaler.fit_transform(train_feature)
    # 用刚刚的StandardScaler对象来进行归一化
    X_test = scaler.transform(test_feature)
    clf = KNeighborsClassifier()
    clf.fit(X, train_label)
    return clf.predict(X_test)
