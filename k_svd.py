################################################################################################
#
#   Date: 2019.11.25
#   -------------------------
#   算法求解思路为交替迭代的进行稀疏编码和字典更新两个步骤.
#   K-SVD在构建字典步骤中，K-SVD不仅仅将原子依次更新，对于原子对应的稀疏矩阵中行向量也依次进行了修正.
#   不像MOP，K-SVD不需要对矩阵求逆，而是利用SVD数学分析方法得到了一个新的原子和修正的系数向量.
#
################################################################################################

import numpy as np
from sklearn import linear_model
import scipy.misc
from matplotlib import pyplot as plt


class KSVD(object):
    def __init__(self,
                 n_components,
                 max_iter=30,
                 tol=1e-6,
                 n_nonzero_coefs=None):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        初始化字典矩阵
        """
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            # np.nonzero 得到矩阵非0元素位置
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            d[:, i] = 0
            # np.dot 矩阵乘法
            r = (y - np.dot(d, x))[:, index]
            # 矩阵的 奇异值 分解
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
        KSVD迭代过程
        """
        self._initialize(y)
        for i in range(self.max_iter):
            # omp算法
            x = linear_model.orthogonal_mp(
                self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            # 求范数
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            if e < self.tol:
                break
            # (300, 512) (512, 512)
            print('item', i, ': ', x.shape, y.shape)
            self._update_dict(y, self.dictionary, x)

        self.sparsecode = linear_model.orthogonal_mp(
            self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode


if __name__ == '__main__':
    im_ascent = scipy.misc.ascent().astype(np.float)
    ksvd = KSVD(300)
    dictionary, sparsecode = ksvd.fit(im_ascent)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_ascent)
    plt.subplot(1, 2, 2)
    plt.imshow(dictionary.dot(sparsecode))
    plt.show()
