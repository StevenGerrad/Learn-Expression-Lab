################################################################################
#
# 读取.pgm: https://blog.csdn.net/quiet_girl/article/details/80904471
#
################################################################################

from PIL import Image
import os
from matplotlib import pyplot as plt
from sklearn import linear_model
import cv2
import numpy as np
import random
import copy

img_path = "./YaleB/"
# n_components: 字典所含原子个数（字典的列数）
n_components = 38 - 1
'''
运行K-SVD算法： 
 字典𝐷大小：64 × 441 
 稀疏度：𝑆 = 10 
 稀疏编码算法：OMP 
'''

select_train_num = 11000


class KSVD(object):
    def __init__(self,
                 n_components=441,
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
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        初始化字典矩阵
        """
        # u, s, v = self._svd(y)
        # print("[_initialize]->u,s,v: ", u.shape, s.shape, v.shape)
        # self.dictionary = u[:, :self.n_components]
        self.dictionary = np.random.rand(8 * 8, self.n_components)

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            # np.nonzero 得到矩阵非0元素位置
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            # 将第i列清空
            d[:, i] = 0
            # E := Y - D X = Y - Σ(l≠j) d_l X
            r = (y - np.dot(d, x))[:, index]
            # 矩阵的 奇异值 分解
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def _svd(self, a):
        '''
        ASK: X = US(V.T)
        https://www.cnblogs.com/endlesscoding/p/10058532.html
        '''
        # --- 特征值分解
        # 1.数据必需先转为浮点型，否则在计算的过程中会溢出，导致结果不准确
        a = a / 255.0
        # 2.计算特征值和特征向量
        s, u = np.linalg.eigh(a.dot(a.T))
        # --- 计算右奇异矩阵
        # 1.降序排列后，逆序输出
        idx = np.argsort(s)[::-1]
        # 2.将特征值对应的特征向量也对应排好序
        s = np.sort(s)[::-1]
        u = u[:, idx]
        # 3.计算奇异值矩阵的逆
        s = np.sqrt(s)
        s_inv = np.linalg.inv(np.diag(s))
        # 4.计算右奇异矩阵
        v = s_inv.dot((u.T).dot(a))
        # 分别为左奇异矩阵，所有奇异值，右奇异矩阵。
        return u, s, v

    def omp(self, B, y, eps=.1, max_iter=90):
        """
        https://blog.csdn.net/theonegis/article/details/78230737
        """
        r = y
        sparse_idx = []
        sparse_min_w = None
        i = 0
        while True:
            i += 1
            # 计算各个原子对y的贡献值，并选择绝对值最大的
            max_r_idx = np.argmax(np.abs(np.dot(r, B)))
            sparse_idx.append(max_r_idx)
            # 估计线性模型中的系数：a=np.linalg.lstsq(x,b),有b=a*x
            sparse_min_w = np.linalg.lstsq(B[:, sparse_idx], y, rcond=None)[0]
            # 更新残差r_t=y − A_t x^t
            r = y - np.dot(B[:, sparse_idx], sparse_min_w)
            err = np.linalg.norm(r, ord=2)
            # print('[omp]->iter: {}, ->err={}'.format(i, err))
            if err < eps or i >= max_iter:
                break
        print("[omp]->times: ", i, ",->err: ", np.linalg.norm(r, ord=2))
        min_w = np.zeros(B.shape[1], dtype=np.float32)
        for idx, val in enumerate(sparse_idx):
            min_w[val] += sparse_min_w[idx]
        # min_w[sparse_idx] = sparse_min_w
        return min_w

    def comp_omp(self, B, y):
        # x_j = self.omp(self.dictionary, y[:, j])
        r = y
        sparse_idx = []
        sparse_min_w = None

        # 计算各个原子对y的贡献值，并选择绝对值最大的
        max_r_idx = np.argmax(np.abs(np.dot(r, B)))
        sparse_idx.append(max_r_idx)
        # 估计线性模型中的系数：a=np.linalg.lstsq(x,b),有b=a*x
        sparse_min_w = np.linalg.lstsq(B[:, sparse_idx], y, rcond=None)[0]
        # 更新残差r_t=y − A_t x^t
        r = y - np.dot(B[:, sparse_idx], sparse_min_w)

        # err = np.linalg.norm(r, ord=2)
        # print("[omp]->i: ", i, ",->err: ", np.linalg.norm(r, ord=2))
        # print('误差', np.linalg.norm(r, ord=2))

        min_w = np.zeros(B.shape[1], dtype=np.float32)
        # 返回 enumerate(枚举) 对象
        for idx, val in enumerate(sparse_idx):
            min_w[val] += sparse_min_w[idx]

        return min_w

    def iter_omp(self, B, y):
        '''
        将 y 拆解按每列更新
        '''
        global select_train_num
        x = np.zeros((self.n_components, select_train_num))
        if (y.ndim == 2):
            for j in range(y.shape[1]):
                # min_w[sparse_idx] = sparse_min_w
                x[:, j] = self.comp_omp(B, y[:, j])
            return x
        else:
            return self.comp_omp(B, y)

    def run2(self, y):
        '''
        omp 算法2，每次迭代，最后更新字典
        '''
        self._initialize(y)

        global select_train_num
        x = np.zeros((self.n_components, select_train_num))
        for j in range(y.shape[1]):
            print("->iter: ", j, end=' ')
            x_j = self.omp(self.dictionary, y[:, j], eps=0.001)
            x[:, j] = x_j

        # 求范数
        e = np.linalg.norm(y - np.dot(self.dictionary, x))
        self._update_dict(y, self.dictionary, x)
        # 最后一次omp返回结果
        self.sparsecode = np.zeros((self.n_components, select_train_num))
        for j in range(y.shape[1]):
            x_j = self.omp(self.dictionary, y[:, j])
            self.sparsecode[:, j] = x_j
        return self.dictionary, self.sparsecode

    def run(self, y):
        self._initialize(y)
        for i in range(self.max_iter):
            # omp算法--1 sklearn 的omp算法包，用以对比分析
            x = linear_model.orthogonal_mp(
                self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)

            # omp算法--3
            # x = self.iter_omp(self.dictionary, y)

            # 求范数
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            print("[run]->i: ", i, ", ->e: ", e)
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)
        self.sparsecode = linear_model.orthogonal_mp(
            self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        # self.sparsecode = self.iter_omp(self.dictionary, y)
        return self.dictionary, self.sparsecode


class Divide:
    def __init__(self, b_w, b_h):
        '''
        b_w: block width
        b_h: block height
        '''
        self.block_width = b_w
        self.block_height = b_h

    def encode(self, mat):
        (W, H) = mat.shape
        # (192, 168)->(24,21)
        w_len = int(W / self.block_width)
        h_len = int(H / self.block_height)
        res = np.zeros((self.block_width * self.block_height, w_len * h_len))
        for i in range(h_len):
            for j in range(w_len):
                temp = mat[j * self.block_width:(j + 1) * self.block_width,
                           i * self.block_height:(i + 1) * self.block_height]
                temp = temp.reshape(self.block_width * self.block_height)
                res[:, i * w_len + j] = temp
        return res

    def decode(self, mat, W, H):
        '''
        mat.shape should be ( block_width*block_height, ~ = 24*21 )
        '''
        w_len = int(W / self.block_width)
        h_len = int(H / self.block_height)
        mat = mat.reshape(self.block_width * self.block_height, w_len * h_len)
        
        res = np.zeros((W, H))
        for i in range(h_len):
            for j in range(w_len):
                temp = mat[:, i * w_len + j]
                temp = temp.reshape(self.block_width, self.block_height)
                res[j * self.block_width:(j + 1) * self.block_width,
                    i * self.block_height:(i + 1) * self.block_height] = temp
        return res


def read_img():
    '''
    人脸数据集中选取的11000个8 × 8的图像 块（各个部位）
    每一个8 × 8的图像块，是64 × 11000的矩阵𝑌中的一列
    '''
    src_img_w = 192
    src_img_h = 168

    # dataset = np.zeros((38,192,168), np.float)
    dataset = np.zeros((src_img_w * src_img_h, 38), np.float)
    cnt_num = 0
    img_list = sorted(os.listdir(img_path))
    os.chdir(img_path)
    for img in img_list:
        if img.endswith(".pgm"):
            # print(img.size)
            gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            # 分块提取 8*8 的 block 汇成矩阵
            # gray_img.reshape(src_img_w * src_img_h, 1)
            divide = Divide(b_w=8, b_h=8)
            gray_img = divide.encode(gray_img)
            # gray_img = cv2.resize(gray_img, (src_img_w, src_img_h),interpolation=cv2.INTER_AREA)
            dataset[:, cnt_num] = gray_img.reshape(src_img_w * src_img_h, )
            cnt_num += 1
    train_data = dataset.reshape(8 * 8, 24 * 21 * 38)
    idx = random.sample(list(range(24 * 21 * (38 - 1))), select_train_num)
    train_data = train_data[:, idx]
    return train_data, dataset[:, -1]


def test_result(B, eva_B):
    '''
    误差计算: 
    error_rate = sqrt(||B - ~B||_2_F / 64)
    '''
    return (np.linalg.norm(B - eva_B))**0.5


if __name__ == "__main__":
    # 实验图像：一张人脸图像，由594个块组成的（不在11000 个训 练数据集中）
    # 1.对于每个8 × 8的图像块𝐵：2.在随机位置，随机删除一部分像素
    train_data, test_data = read_img()
    # 按块处理
    divide = Divide(b_w=8, b_h=8)
    missed_img = copy.deepcopy(test_data)
    missed_img = missed_img.reshape(8 * 8, 21 * 24)
    for j in range(21 * 24):
        # missed_img[:, j][random.randint(0, 64 - 1)] = 0
        missed_img[:, j][random.sample(list(range(0, 64)), 10)] = 0
    plt.subplot(1, 2, 1), plt.imshow(divide.decode(test_data, 192, 168))
    plt.subplot(1, 2, 2), plt.imshow(divide.decode(missed_img, 192, 168))
    plt.show()
    cv2.imwrite('test_data.png',divide.decode(test_data.reshape(8 * 8, 24 * 21), 192, 168))
    cv2.imwrite('missed_img.png', divide.decode(missed_img, 192, 168))

    print("[train_data]: ", train_data.shape, "[test_data]: ", test_data.shape)

    # show the dictionary
    # plt.imshow(divide.decode(train_data, 110 * 8, 100 * 8)),plt.show()

    # 3.分别基于学习到的K-SVD字典、直接构建的Haar字典、 DCT字典，使用OMP算法，获得受损图像的稀疏表达
    k_svd = KSVD()
    # run(): iter_omp / sklearn , run2(): omp
    # dictionary, sparsecode = k_svd.run(train_data)
    dictionary, sparsecode = k_svd.run2(train_data)

    plt.subplot(1, 2, 1), plt.imshow(divide.decode(train_data, 110 * 8, 100 * 8))
    plt.subplot(1, 2, 2), plt.imshow(divide.decode(dictionary, 21 * 8, 21 * 8))
    plt.show()

    # 4.块𝐵的系数矩阵表示为𝑋_𝐵
    test_data = test_data.reshape(8 * 8, 21 * 24)
    # 对于缺失部分进行剔除 TODO：图像原本就有0值怎么办
    x = np.zeros((441, 21 * 24))
    for j in range(21 * 24):
        idx = []
        for i in range(missed_img.shape[0]):
            if missed_img[i, j] == 0:
                idx.append(i)
        # print(len(idx), end=' ')
        y = copy.deepcopy(missed_img[:, j])
        B = copy.deepcopy(dictionary)
        y = np.delete(y, idx)
        B = np.delete(B, idx, axis=0)
        # print(y.shape, B.shape, end=' ')
        # 三种不同的omp策略
        # x_j = linear_model.orthogonal_mp(B, y, n_nonzero_coefs=None)
        print("->iter: ", j, end=' ')
        x_j = k_svd.omp(B, y, eps=.1)
        # x_j = k_svd.iter_omp(B, y)
        x[:, j] = x_j

    print("[x with missed_img]->shape: ", x.shape)
    plt.bar([1, 3, 5, 7, 9], x, label='graph 1')
    plt.show()
    # 5.重构的块 ~𝐵 = 𝐷 𝑋_𝐵
    # 6.重构误差：
    result = dictionary.dot(x)
    print("[test_result]: ", test_result(test_data, result))
    print("[missed_img]: ", test_result(missed_img, result))
    # cv2.imwrite('result.png', divide.decode(result, 192, 168))

    # show the image
    plt.subplot(1, 3, 1), plt.imshow(divide.decode(test_data, 192, 168))
    plt.subplot(1, 3, 2), plt.imshow(divide.decode(missed_img, 192, 168))
    plt.subplot(1, 3, 3), plt.imshow(divide.decode(result, 192, 168))
    plt.show()
