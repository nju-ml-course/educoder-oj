from PIL import Image
import numpy as np
from sklearn.mixture import GaussianMixture

# 读取road.jpg到im变量中
im = Image.open('./step3/image/test.jpg')

# 将im转换成ndarray
img = np.array(im)
# 将img变形为[-1, 3]的shape，并保存至img_reshape
img_reshape = img.reshape(-1, 3)

# 实例化一个将数据聚成3个簇的高斯混合聚类器
gmm = GaussianMixture(3)
# 将数据传给fit函数，fit函数会计算出各个高斯分布的参数和响应系数
gmm.fit(img_reshape)
# 对数据进行聚类，簇标记为0 1 2(因为gmm对象想要聚成3个簇)
pred = gmm.predict(img_reshape)

img_reshape[pred == 0, :] = [255, 255, 0]  # 黄色
img_reshape[pred == 1, :] = [0, 0, 255]  # 蓝色
img_reshape[pred == 2, :] = [0, 255, 0]  # 绿色
im = Image.fromarray(img.astype('uint8'))
# 将im保存为new_road.jpg
im.save('./step3/dump/result.jpg')
