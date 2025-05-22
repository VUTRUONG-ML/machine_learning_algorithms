from __future__ import print_function
import numpy as np
from time import time # de so sanh thoi gian chay 
d, N = 1000, 1000 # Chieu, so diem training 
M = 100
Z = np.random.randn(M, d)
X = np.random.randn(N, d).astype(np.float32) # tao mang 2 chieu voi cac so ngau nhien có mean = 0, std = 1
z = np.random.randn(d).astype(np.float32) 

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

# Tu moi diem trong tap hop nay toi moi diem trong tap hop kia 
def dist_ss_0(Z, X):
    M = Z.shape[0]
    N = X.shape[0]
    res = np.zeros((M, N))
    for i in range(M):
        res[i] = dist_ps_fast(Z[i], X)
    return res 

def dist_ss_fast(Z, X):
    Z2 = np.sum(Z*Z, 1) # axis = 1 , tinh
    X2 = np.sum(X*X, 1) # axis = 1 , tinh tong
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2 * Z @ X.T

t1 = time()
D3 = dist_ss_0(Z, X)
print('half fast set2set running time:', time() - t1,'s')
t1 = time()
D4 = dist_ss_fast(Z, X)
print('fast set2set running time', time() - t1, 's')
print('Result difference:', np.linalg.norm(D3 - D4))