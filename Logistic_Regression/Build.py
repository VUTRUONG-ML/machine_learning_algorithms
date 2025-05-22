import numpy as np


def Sigmoid(S):
    return 1/(1 + np.exp(-S))

def prob(w, X):
    # X: 2d shape(N, d): N data point, d feature
    # w: 1d shape(d)
    return Sigmoid(X @ w)

def loss(w, X, y, lam):
    z = prob(w, X)
    return -np.mean(y * np.log(z) + (1 - y)*np.log(1 - z)) + .5*lam/X.shape[0]*np.sum(w*w)

def logistic_regression(w_init, X, y, lamda = 0.001, lr = 0.1, nloop = 2000):
    # nloop : number of loop
    N, d = X.shape[0], X.shape[1]
    w = w_old = w_init
    loss_hist = [loss(w_init, X, y, lamda)] # Luu lai cac gia tri cua ham J(w)
    for it in range(nloop):
        mix_ids = np.random.permutation(N) # Trộn danh sách các điểm dữ liệu
        for i in mix_ids:
            xi = X[i]
            yi = y[i]
            zi = Sigmoid(xi @ w)
            w = w - lr*((zi - yi)*xi + lamda*w)
        loss_hist.append(loss(w, X, y, lamda))
        if np.linalg.norm(w - w_old)/d < 1e-6: 
            break
        w_old = w
    return w, loss_hist

# Ngưỡng quyết định xác suất có thể thay đổi, thres 
def predict(w, X, thres = 0.5):
    res = np.zeros(X.shape[0])
    # cho các chỉ số nào mà có xác suất lớn hơn thres thì phân vào class 1 và ngược lại, vì thế nên mình có thể điều chỉnh thres 
    res[np.where(prob(w, X) > thres)[0]] = 1
    return res

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)

w_init = np.random.randn(Xbar.shape[1])
lam = 0.0001
w, loss_hiss = logistic_regression(w_init, Xbar, y, lam, 0.05, 500)
print(w)
print(loss(w, Xbar, y, lam))

# Tính xác suât bằng cách y = w*x + w0 
