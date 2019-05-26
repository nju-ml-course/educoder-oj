import pandas as pd
from sklearn.linear_model.perceptron import Perceptron
import csv

# 获取训练数据
train_data = pd.read_csv('./step2/train_data.csv')
# 获取训练标签
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
# 获取测试数据
test_data = pd.read_csv('./step2/test_data.csv')
clf = Perceptron(max_iter=1e5)
clf.fit(train_data, train_label)
result = clf.predict(test_data)

pd.DataFrame({'result': result}).to_csv('./step2/result.csv', index=False)
