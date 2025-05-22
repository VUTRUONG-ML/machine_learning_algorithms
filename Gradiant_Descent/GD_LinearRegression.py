import numpy as np
X = np.random.rand(1000)
y = 4 + 3 * X + .5*np.random.rand(1000) # y = 4 + 3X + noise    

Xbar = np.ones((1000, 1))
Xbar = np.concatenate((Xbar, X.reshape(-1,1)), axis=1)

def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T @ (Xbar @ w - y)

def cost(w):
    N = Xbar.shape[0]
    return 1/(2*N) * np.linalg.norm(y - Xbar @ w)**2

def myGD(w_init, grad, learning_rate):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - learning_rate*grad(w[-1])
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)

w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 1)
print(f'Sol found by GD: w = {w1[-1][0, 0]:.4f}, w1 = {w1[-1][1, 0]:.4f}, \nafter {it1 + 1} iteration.')