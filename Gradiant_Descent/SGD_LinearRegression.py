import numpy as np

np.random.seed(21)
X = np.random.rand(1000, 1)
Y = 4 + 3*X + .5*np.random.rand(1000, 1)

Xbar = np.hstack((np.ones((X.shape[0], 1)), X))

def grad_sgd(w, xi, yi):
    return (xi.T @ (xi @ w - yi))

def SGD(w0, grad_sgd, learning_rate, Xbar, y):
    w = [w0]
    N = Xbar.shape[0]
    for it in range(100):
        i = np.random.randint(0, N)
        xi = Xbar[i : i + 1] # lay mot hàng từ  Xbar
        yi = y[i : i + 1]
        w_new = w[-1] - learning_rate * grad_sgd(w[-1], xi, yi)
        if(np.linalg.norm(w_new - w[-1])) < 1e-3:
            break
        w.append(w_new)
    return w, it

w_init = np.array([[2], [1]])
learning_rate = 0.01

# Chạy SGD
w_sgd, num_iter = SGD(w_init, grad_sgd, learning_rate, Xbar, Y)

# Kết quả
print("Số vòng lặp:", num_iter)
print("Trọng số tìm được (SGD):", w_sgd[-1].flatten())

# Mini-batch Gradient Descent.
def grad_mbgd(w, x, y):
    N = x.shape[0]
    return (1/N) * (x.T @ (x @ w - y))

def grad_mbgd(w0, grad_mbgd, learning_rate, Xbar, Y, batch_size):
    w = [w0]
    N = Xbar.shape[0]
    for it in range(100):
        index = np.random.choice(Xbar.shape[0], batch_size, replace=False)
        x = Xbar[index]
        y = Y[index]
        w_new = w[-1] - learning_rate * grad_mbgd(w[-1], x, y)
        if(np.linalg.norm(w_new - w[-1])) < 1e-3:   
            break
        w.append(w_new)
    return w 