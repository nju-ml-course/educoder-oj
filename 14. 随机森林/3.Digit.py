from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sklearn.datasets as db


def digit_predict(train_image, train_label, test_image):
    """
    实现功能：训练模型并输出预测结果
    :param train_image: 包含多条训练样本的样本集，类型为ndarray,shape为[-1, 8, 8]
    :param train_label: 包含多条训练样本标签的标签集，类型为ndarray
    :param test_image: 包含多条测试样本的测试集，类型为ndarry
    :return: test_image对应的预测标签，类型为ndarray
    """
    X = np.reshape(train_image, newshape=(-1, 64))
    clf = RandomForestClassifier(n_estimators=500, max_depth=10)
    clf.fit(X, y=train_label)
    return clf.predict(test_image)


data = db.load_digits()
