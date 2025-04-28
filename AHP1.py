#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def get_w(array):
    row = array.shape[0]  # 计算出阶数
    eigenvalues, eigenvectors = np.linalg.eig(array)
    max_idx = np.argmax(eigenvalues)
    max_eigen = eigenvalues[max_idx].real
    eigen = eigenvectors[:, max_idx].real
    eigen = eigen / eigen.sum()
    CI = (max_eigen - row) / (row - 1)
    CR = CI / RI_dict[row]
    if CR < 0.1:
        # print(round(CR, 3))
        # print('满足一致性')
        # print(np.max(w))
        # print(sorted(w,reverse=True))
        # print(max_max)
        # print('特征向量:%s' % w)
        return eigen
    else:
        print(round(CR, 3))
        print('不满足一致性，请进行修改')


def main(array):
    if type(array) is np.ndarray:
        return get_w(array)
    else:
        print('请输入numpy对象')


if __name__ == '__main__':
    # 由于地方问题，矩阵我就写成一行了
    e = np.array([[1, 2, 7, 5, 5], [1 / 2, 1, 4, 3, 3], [1 / 7, 1 / 4, 1, 1 / 2, 1 / 3], [1 / 5, 1 / 3, 2, 1, 1], [1 / 5, 1 / 3, 3, 1, 1]])
    a = np.array([[1, 1 / 3, 1 / 8], [3, 1, 1 / 3], [8, 3, 1]])
    b = np.array([[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]])
    c = np.array([[1, 1, 3], [1, 1, 3], [1 / 3, 1 / 3, 1]])
    d = np.array([[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]])
    f = np.array([[1, 4, 1 / 2], [1 / 4, 1, 1 / 4], [2, 4, 1]])
    e = main(e)
    a = main(a)
    b = main(b)
    c = main(c)
    d = main(d)
    f = main(f)
    try:
        res = np.array([a, b, c, d, f]) # 5*3 5 代表性质，3代表人数（在这里改）
        ret = (np.transpose(res) * e).sum(axis=1) # 每一个候选人的所有性质(3*5)乘以相应的权重,得到一个3*5的矩阵即每一个候选人的性质比重，再把行数加起来
        print(ret)
    except TypeError:
        print('数据有误，可能不满足一致性，请进行修改')