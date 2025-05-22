import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.svm import SVC
np.random.seed(31)

# Số điểm dữ liệu
num_samples = 200
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]


N = 10 
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N) # shape (N, 2) 
X = np.concatenate((X0, X1), axis=0) # shape (2N, 2)
y = np.concatenate((np.ones(N), -np.ones(N)), axis=0) # shape (2N,)

# solving the dual problem (variable: lambda)
V = np.concatenate((X0, -X1), axis=0) # shape (2N, 2) same with x.y 
K = matrix(V @ V.T) # this is K in the paper # P in quadratic programing
p = matrix(-np.ones((2*N, 1))) # objective function 1/2 lambda^T*K*lambda - 1^T*lambda

# build A, b, G, h
G = matrix(-np.eye(2*N)) # Ma trận -I để đảm bảo λ >= 0
h = matrix(np.z eros((2*N, 1))) # Vector 0 để biểu diễn λ >= 0 | G.lambda <= h 
A = matrix(y.reshape(1, -1)) # A.lambda = b | A = y.reshape 
b = matrix(np.zeros((1, 1))) # Giá trị b = 0
solvers.options['show_pregress'] = False    
sol = solvers.qp(K, p, G, h, A, b)
l = np.array(sol['x']) # solution lambda

# calculate w and b
w = np.sum(l * y[:, None] * X, axis=0) # shape (2,)
S = np.where(l > 1e-8)[0] # support set, 1e-8 to avoid small value of l.
b = np.mean(y[S].reshape(-1, 1) - X[S,:].dot(w))
print('Number of suport vectors = ', S.size)
print('w = ', w.T)
print('b = ', b)

# Build with skleanr
model = SVC(kernel= 'linear', C=1e5)
model.fit(X, y)
w_s = model.coef_
b_s = model.intercept_
print('ws = ', w_s.T)
print('bs = ', b_s)
# Vẽ dữ liệu và đường SVM
plt.figure(figsize=(8,6))
plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='Class 1')
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class -1')
plt.scatter(X[S, 0], X[S, 1], s=100, edgecolors='k', facecolors='none', label='Support Vectors')

# Vẽ đường quyết định
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
x_values = np.linspace(x_min, x_max, 100)
y_values = (-w[0] / w[1]) * x_values - b / w[1]

# Vẽ đường lề (margin)
margin1 = (-w[0] / w[1]) * x_values - (b - 1) / w[1]
margin2 = (-w[0] / w[1]) * x_values - (b + 1) / w[1]

plt.plot(x_values, y_values, 'k-', label='Decision Boundary')
plt.plot(x_values, margin1, 'k--', label='Margin')
plt.plot(x_values, margin2, 'k--')

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("SVM Decision Boundary")
plt.legend()
plt.show()