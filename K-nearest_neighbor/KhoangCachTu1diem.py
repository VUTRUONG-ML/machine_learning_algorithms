from __future__ import print_function
import numpy as np
from time import time # de so sanh thoi gian chay 
d, N = 1000, 1000 # Chieu, so diem training 
X = np.random.randn(N, d).astype(np.float32) # tao mang 2 chieu voi cac so ngau nhien có mean = 0, std = 1
z = np.random.randn(d).astype(np.float32) # tao mang 1 chieu voi cac so ngau nhien có mean = 0

# Tinh khoang cach giua hai diem
def dist_pp(z, x):
    d = z - x.reshape(z.shape)  # x va z phai cung kich co 
    return np.sum(d*d)

# Tinh khoang cach tu mot diem toi moi diem trong tap hop, tính cách cơ bản nhất 
def dist_ps_naive(z, X):
    N = X.shape[0]
    res = np.zeros((1, N))
    for i in range(N):
        res[0][i] = dist_pp(z, X[i])
    return res 

# Tinh khoan cach nhu tren, nhung bang cach nhanh hon 
def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1) # axis = 1 , tinh tong theo hang 
    z2 = np.sum(z*z)
    return X2 + z2 - 2 * X @ z

t1 = time()
D1 = dist_ps_naive(z, X)
print('half fast set2set running time:', time() - t1, 's')
t1 = time()
D2 = dist_ps_fast(z, X)
print('fast set2set running time', time() - t1, 's')
print('Result difference:', np.linalg.norm(D1 - D2))