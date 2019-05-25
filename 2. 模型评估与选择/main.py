import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def confusion_matrix(y_true, y_predict):
    '''
    构建二分类的混淆矩阵，并将其返回
    :param y_true: 真实类别，类型为ndarray
    :param y_predict: 预测类别，类型为ndarray
    :return: 二维list或shape为(2, 2)的ndarray
    '''
    ans = [[0, 0], [0, 0]]
    for i in range(len(y_predict)):
        ans[y_true[i]][y_predict[i]] += 1
    return np.array(ans)


def precision_score_(y_true, y_predict):
    '''
    计算精准率并返回
    :param y_true: 真实类别，类型为ndarray
    :param y_predict: 预测类别，类型为ndarray
    :return: 精准率，类型为float
    '''
    arr = confusion_matrix(y_true=y_true, y_predict=y_predict)
    return arr[1][1] / (arr[1][1] + arr[0][1])


def recall_score_(y_true, y_predict):
    '''
    计算召回率并召回
    :param y_true: 真实类别，类型为ndarray
    :param y_predict: 预测类别，类型为ndarray
    :return: 召回率，类型为float
    '''
    arr = confusion_matrix(y_true=y_true, y_predict=y_predict)
    return arr[1][1] / (arr[1][1] + arr[1][0])


def calAUC(prob, labels):
    '''
    计算AUC并返回
    :param prob: 模型预测样本为Positive的概率列表，类型为ndarray
    :param labels: 样本的真实类别列表，其中1表示Positive，0表示Negtive，类型为ndarray
    :return: AUC，类型为float
    '''
    M = len([_ for _ in labels if _ == 1])
    N = len(labels) - M

    # i of the sorted arr,labels
    rank = []
    for i, formal_index in enumerate(np.argsort(prob)):
        rank_item = i + 1
        rate = prob[formal_index]
        if labels[formal_index] == 1:
            if formal_index > 0 and prob[formal_index - 1] == rate and labels[formal_index - 1] == 0:
                rank.append(rank_item - 0.5)
            elif formal_index < len(prob) - 1 and prob[formal_index + 1] == rate and labels[formal_index + 1] == 0:
                rank.append(rank_item + 0.5)
            else:
                rank.append(rank_item)
    return (np.sum(rank) - (M + 1) * M / 2) / (M * N)


def classification_performance(y_true, y_pred, y_prob):
    '''
    返回准确度、精准率、召回率、f1 Score和AUC
    :param y_true:样本的真实类别，类型为`ndarray`
    :param y_pred:模型预测出的类别，类型为`ndarray`
    :param y_prob:模型预测样本为`Positive`的概率，类型为`ndarray`
    :return:
    '''
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred), \
           f1_score(y_true, y_pred), roc_auc_score(y_true, y_prob)


