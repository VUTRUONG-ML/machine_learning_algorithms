import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
np.random.seed(32)

means = [[0, 5], [5, 0]]
cov0 = [[4,3], [3, 4]]
cov1 = [[3,1], [1, 1]]
N0, N1 = 50, 40 
N = N0 + N1
X0 = np.random.multivariate_normal(means[0], cov0, N0) # each row is a data point
X1 = np.random.multivariate_normal(means[1], cov1, N1) 

# Build S_B
m0 = np.mean(X0.T, axis = 1, keepdims = True) # vector trung bình của từng hàng 
m1 = np.mean(X1.T, axis = 1, keepdims = True)

a = (m0 - m1)
S_B = a @ a.T

# Build S_W 
A = X0.T - np.tile(m0, (1, N0)) # np.tile(m0, (1, N0)) is equivalent to m0.T @ np.ones((1, N0)) = là ma trận trung bình của X0
B = X1.T - np.tile(m1, (1, N1))
S_W = A @ A.T + B @ B.T 

_, W = np.linalg.eig(np.linalg.inv(S_W ) @ S_B) # gia tri rieng, vector rieng 

w = W[:, 0] # vector riêng cột đầu tiên, hướng chiếu quan trọng nhất - để giá trị riêng lớn nhất   
print('W = ', w)

# Ap dung sklearn 
X = np.concatenate((X0, X1))
y = np.array([0]*N0 + [1]*N1)
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print('W_sklearn = ', clf.coef_[0]/np.linalg.norm(clf.coef_))