# encoding=utf8
import os
from sklearn.neural_network import MLPClassifier
import pandas as pd

if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

# ********* Begin *********#
# 获取训练数据
train_data = pd.read_csv('./step2/train_data.csv')
# 获取训练标签
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
# 获取测试数据
test_data = pd.read_csv('./step2/test_data.csv')

mlp = MLPClassifier(solver='lbfgs', max_iter=100,
                    alpha=1e-5, hidden_layer_sizes=(5, 10, 3))
mlp.fit(train_data, train_label)
result = mlp.predict(test_data)

result = pd.DataFrame(result, columns=['result'])

result.to_csv('./step2/result.csv', index=False)

# ********* End *********#
