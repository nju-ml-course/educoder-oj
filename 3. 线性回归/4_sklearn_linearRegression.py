# encoding=utf8
# ********* Begin *********#
from sklearn.linear_model import LinearRegression
import pandas as pd

# 获取训练数据
train_data = pd.read_csv('./step3/train_data.csv')
# 获取训练标签
train_label = pd.read_csv('./step3/train_label.csv')
train_label = train_label['target']
# 获取测试数据
test_data = pd.read_csv('./step3/test_data.csv')

model = LinearRegression(normalize=True)
model.fit(train_data, train_label)
test_y = model.predict(test_data)

pd.DataFrame(test_y, columns=['result']).to_csv('./step3/result.csv')

# ********* End *********#
