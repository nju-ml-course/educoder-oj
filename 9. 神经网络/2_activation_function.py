# encoding=utf8

def relu(x):
    '''
    x:负无穷到正无穷的实数
    '''
    # ********* Begin *********#
    if x <= 0:
        return 0
    else:
        return x
    # ********* End *********#
