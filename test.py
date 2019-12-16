'''

import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(a)
a = a.reshape(3, 4)
print(a)
b = np.zeros((12, 12), np.float)
for i in range(12):
    b[:, i] = a.reshape(12, )
print(b)
a = b[:, -1]
print(a)

'''

# 第一次生成的图忘了decode了

'''

import cv2
from matplotlib import pyplot as plt
import numpy as np


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
        mat = mat.reshape(self.block_width * self.block_height, 24 * 21)
        w_len = int(W / self.block_width)
        h_len = int(H / self.block_height)
        res = np.zeros((W, H))
        for i in range(h_len):
            for j in range(w_len):
                temp = mat[:, i * w_len + j]
                temp = temp.reshape(self.block_width, self.block_height)
                res[j * self.block_width:(j + 1) * self.block_width,
                    i * self.block_height:(i + 1) * self.block_height] = temp
        return res


if __name__ == '__main__':
    img = cv2.imread('YaleB/result.png', cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    divide = Divide(8, 8)
    img = divide.decode(img.reshape(8 * 8, 24 * 21), 192, 168)
    plt.imshow(img)
    plt.show()

'''