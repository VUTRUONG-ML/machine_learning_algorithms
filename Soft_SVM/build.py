import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from cvxopt import matrix, solvers

np.random.seed(42)
means = [[2,2],[4,2]]
cov = [[.7, 0],[0,.7]]

N = 20 
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1)) # shape (40, 2)
y = np.concatenate((np.ones(N), -np.ones(N))) # shape (40,)

C = 100

# Build with sklearn
clf = SVC(kernel='linear', C=C)
clf.fit(X, y)
W_sklearn = clf.coef_.reshape(-1, 1)
b_sklearn = clf.intercept_[0]

print(W_sklearn, b_sklearn)

#Build with CVXOPT
#Build Q 
V = np.concatenate((X0, -X1), axis = 0) # V[n,:] = y[n]*X[n] - Tại vì khi nhân với y thì đoạn x0 sẽ dương còn đoạn x1 sẽ âm 
Q = matrix(V @ V.T)
p = matrix(-np.ones((2*N,1))) 
#Build G, h, A, b 
G = matrix(np.vstack((-np.eye(2*N), np.eye(2*N)))) # 2 điều kiện -lamda <= 0 và -lamda <= C 
h = np.vstack((np.zeros((2*N,1)), C*np.ones((2*N, 1)))) # G.lamda <= h mà tại vì  0 <=lamda <= C
h = matrix(h)
A = matrix(y.reshape((-1, 2*N))) # A @ lamda = 0 tương đương với điều kiện tổng y*lamda = 0 
b = matrix(np.zeros((1,1)))

solvers.options['show_progress'] = False
sol = solvers.qp(Q, p, G, h, A, b)

l = np.array(sol['x']).reshape(2*N)

# support set 
S = np.where(l > 1e-5)[0]
S2 = np.where(l < .999*C)[0]
# margin set
M = [val for val in S if val in S2] # điểm có lamda > 0 và < C

VS = V[S]           # shape(NS, d)
lS = l[S]           # shape(NS,)
w_dual = lS @ VS    # shape(d,) # w = lamda*y*x 
yM = y[M]           # shape(NM,)
XM = X[M]           # shape(NM,d)
b_dual = np.mean(yM - XM @ w_dual) # tính b dùng lamda trong khoảng (0, C) vì nếu lamda [C, +00] có nghĩa là lamda = C => Muy = 0 => epxilon(điểm sai phạm) != 0 => không có margin
print('w_dual = ', w_dual)
print('b_dual = ', b_dual)


# Build with gradient descent 
lam = 1./C
def loss(X, y, w, b):
    # X(2N, d), w(d,), y(2N,)
    z = X @ w + b # (2N, )
    yz = y*z 
    return (np.sum(np.maximum(0,1 - yz)) + .5*lam*w @ w) / X.shape[0]

def grad(X, y, w, b):
    z = X @ w + b
    yz = y*z
    active_set = np.where(yz <= 1)[0] # consider 1 - y(w.x + b) >= 0 only 
    _yX = - X*y[:, np.newaxis] # reshape y(N,) -> y(N,1), Ở đấy sử dụng dấu '*' thay vì '@' là nó sẽ nhân từng hàng với nhau
    grad_w = (np.sum(_yX[active_set], axis = 0) + lam*w)/X.shape[0]
    grad_b = (-np.sum(y[active_set]))/X.shape[0]
    return (grad_w, grad_b)

def num_grad(X, y, w, b): # Dùng để kiếm tra đạo hàm bên trên có đúng ko, f'(X) gần bằng [f(x + eps) - f(x - eps)]/2eps
    eps = 1e-10 
    gw = np.zeros_like(w)
    gb = 0 
    for i in range(len(w)):
        wp = w.copy()
        wm = w.copy()
        wp[i] += eps 
        wm[i] -= eps 
        gw[i] = (loss(X, y, wp, b) - loss(X, y, wm, b)) / (2*eps)
    gb = (loss(X, y, w, b + eps) - loss(X, y, w, b - eps)) / (2*eps) 
    return (gw, gb)

w = .1*np.random.randn(X.shape[1])
b = np.random.randn()
(gw0, gb0) = grad(X, y, w, b)
(gw1, gb1) = num_grad(X, y, w, b)
# print('grad_w difference = ', np.linalg.norm(gw0 - gw1))
# print('grad_b difference = ', np.linalg.norm(gb0 - gb1))
# Sau bước kiểm tra thì sự khác nhau là rất nhỏ nên grad của mình tính đúng 

def soft_SVM_gd(X, y, w0, b0, eta):
    w = w0 
    b = b0 
    it = 0 
    while it < 10000:
        it = it + 1
        (gw, gb) = grad(X, y, w, b)
        w -= eta*gw 
        b -= eta*gb 
        if(it % 1000 == 0):
            print('iter %d' %it + ' loss: %f' %loss(X, y, w, b))
    return (w, b)

w0 = .1*np.random.randn(X.shape[1])
b0 = .1*np.random.randn()
lr = 0.05

(w_higne, b_higne) = soft_SVM_gd(X, y, w0, b0, lr)
print('w_hinge = ', w_dual)
print('b_hinge = ', b_dual)