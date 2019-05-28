# -*- coding: utf-8 -*-
from sklearn.manifold import Isomap
import isomap as isa
import sklearn.datasets as db



def isomap(data, d, k):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维度
          k(int):最近的k个样本
    output:Z(ndarray):降维后数据
    '''
    # ********* Begin *********#
    iso = Isomap(n_neighbors=k, n_components=d)
    return iso.fit_transform(data)


if __name__ == '__main__':
    ir = db.load_boston()
    X1 = isa.isomap(ir.data[:10], d=2, k=4)
    X2 = isomap(ir.data[:10], d=2, k=4)
    print(X1)
    print(X2)
