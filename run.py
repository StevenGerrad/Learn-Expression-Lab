
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


DATA_DIR = './AR/'
DICT_FILE = './dict.npy'
IMG_SHAPE = (165//2, 120//2)
DICT_GESTURE = [1, 2, 3, 4, 5, 6, 7]
# DICT_GESTURE = [1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20]

def read_flat(path, normalize=False):
	# col = None
	# with open(path, 'rb') as raw:
	# 	raw.seek(0x0F)
	# 	col = np.fromfile(raw, dtype=np.uint8)
	# col = col.astype(np.float32)
	# col = col.reshape(IMG_SHAPE)
	# if normalize:
	# 	for i in range(col.shape[0]):
	# 		col[i] = col[i] * 100. / np.linalg.norm(col[i])
	# return col
	col = cv2.imread(path, 0)
	col = cv2.resize(col, IMG_SHAPE[::-1])
	col = col.reshape([-1]).astype(np.float32)
	if normalize:
		m = np.linalg.norm(col)
		col = col * 100. / m
	return col

def visualize(gray):
	# gray[gray > 255] = 255
	gray = gray.reshape(IMG_SHAPE)
	return gray

def build_dict():
	"""
	A.shape -> [图片数, 行*列]
	"""

	A = []
	if os.path.exists(DICT_FILE):
	# if False:
		print('加载字典...')
		A = np.load(DICT_FILE)
	else:
		print('建立字典...')
		filelist = os.listdir(DATA_DIR)
		for filename in filelist:
			if not filename.endswith('.pgm'):
				continue
			basename = filename.split('.')[0]
			gesture = int(basename.split('-')[-1])
			if gesture not in DICT_GESTURE:
				continue
			col = read_flat(os.path.join(DATA_DIR, filename), normalize=True)
			A.append(col)
		A = np.array(A, dtype=np.float32)
		np.save(DICT_FILE, A)
	print('完成')
	return A

def omp(B, y, eps=1, max_iter=60):
	"""
	w^ = min(L1_norm(w)) s.t. L2_norm(y-B*w)<eps
	w = [x e]
	B = [A I]
	"""

	# print('求解稀疏表达...')
	r = y
	sparse_idx = []
	sparse_min_w = None
	i = 0
	while True:
		i += 1
		# print(i)
		max_r_idx = np.argmax(np.abs(np.dot(B, r)))
		sparse_idx.append(max_r_idx)
		sparse_min_w = np.linalg.lstsq(B[sparse_idx,:].T, y, rcond=None)[0]
		# B1 = B[sparse_idx,:]
		# pinv = np.matmul(np.linalg.inv(np.matmul(B1.T, B1)),B1.T)
		# sparse_min_w = np.matmul(pinv, y)
		r = y - np.dot(B[sparse_idx,:].T, sparse_min_w)
		err = np.linalg.norm(r, ord=2)
		print('第{}次迭代，误差={}'.format(i, err))
		if err < eps or i >= max_iter:
			break
	# print('误差', np.linalg.norm(r, ord=2))
	min_w = np.zeros(B.shape[0], dtype=np.float32)
	for idx, val in enumerate(sparse_idx):
		min_w[val] += sparse_min_w[idx]
	# min_w[sparse_idx] = sparse_min_w
	# print('完成')
	return min_w

def test_rebuild(A, y):
	I = np.identity(A.shape[1], dtype=np.float32) * 100.
	B = np.concatenate([A, I])
	w = omp(B, y, max_iter=200)
	y_r = np.dot(B.T, w)
	w[:-I.shape[0]] = 0.
	y_e = np.dot(B.T, w)

	plt.subplot(1, 4, 1)
	plt.axis('off')
	plt.title('原始图像')
	plt.imshow(visualize(y), cmap='gray')
	plt.subplot(1, 4, 2)
	plt.axis('off')
	plt.title('拟合图像')
	plt.imshow(visualize(y_r), cmap='gray')
	plt.subplot(1, 4, 3)
	plt.axis('off')
	plt.title('修复图像')
	plt.imshow(visualize(y_r - y_e), cmap='gray')
	plt.subplot(1, 4, 4)
	plt.axis('off')
	plt.title('遮挡')
	plt.imshow(visualize(y_e), cmap='gray')
	plt.tight_layout()
	plt.show()

def test_pixel_corruption(A, y):
	I = np.identity(A.shape[1], dtype=np.float32) * 100.
	B = np.concatenate([A, I])
	y_corr = y.copy()
	y_corr[np.random.rand(IMG_SHAPE[0]*IMG_SHAPE[1])>0.1] = 0
	w = omp(B, y_corr, max_iter=200)
	x = w.copy()
	x[-B.shape[1]:] = 0.
	e = w.copy()
	e[: -B.shape[1]] = 0.
	
	plt.subplot(1, 4, 1)
	plt.axis('off')
	plt.title('原始图像')
	plt.imshow(visualize(y), cmap='gray')

	plt.subplot(1, 4, 2)
	plt.axis('off')
	plt.title('损坏图像')
	plt.imshow(visualize(y_corr), cmap='gray')

	plt.subplot(1, 4, 3)
	plt.axis('off')
	plt.title('修复图像')
	plt.imshow(visualize(np.dot(B.T, x)), cmap='gray')

	plt.subplot(1, 4, 4)
	plt.axis('off')
	plt.title('噪声')
	plt.imshow(visualize(np.dot(B.T, e)), cmap='gray')
	
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

	path = os.path.join(DATA_DIR, 'w-014-13.pgm')
	y = read_flat(path, normalize=True)
	A = build_dict()
	
	test_rebuild(A, y)
	# test_pixel_corruption(A, y)

	# 数字图像处理
	# 环境、数据、操作手册、ppt
	# 学号_姓名_大作业_题目.rar