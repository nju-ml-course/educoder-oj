# -*- coding: utf-8 -*-
from sklearn.manifold import MDS


def mds(data, d):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维度
    output:Z(ndarray):降维后数据
    '''
    # ********* Begin *********#
    mds = MDS(d)
    Z = mds.fit_transform(data)

    # ********* End *********#
    return Z





















