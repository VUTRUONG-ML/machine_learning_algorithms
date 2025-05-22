import numpy as np
X = np.random.rand(1000) # shape : (1000,)
y = 4 + 3 * X + .5*np.random.rand(1000) # y = 4 + 3X + noise, shape : (1000,)

Xbar = np.ones((1000, 1))
Xbar = np.concatenate((Xbar, X.reshape(-1,1)), axis=1) # shape : (1000, 2)

def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T @ (Xbar @ w - y) # shape : (2,) theo w 
def grad_sgd(w, xi, yi):
    return xi.T @ (xi @ w - yi) # shape : (2,) theo w
def grad_mbgd(w, xk, yk): 
    k = xk.shape[0]
    return 1/k * xk.T @ ( xk @ w - yk) # shape : (2,) theo w

def cost(w):
    N = Xbar.shape[0]
    return 1/(2*N) * np.linalg.norm(y - Xbar @ w) ** 2

def GRD(w_init, lr):
    w = [w_init]
    for i in range(100):
        w_new = w[-1] - lr*grad(w[-1])
        if(np.linalg.norm(grad(w_new))/ len(w_new) < 1e-6):
            break 
        w.append(w_new)
    return w[-1], i

def GD_momentum(w_init, lr, gm):
    w = [w_init]
    v_old = np.zeros_like(w_init)
    for i in range(100):
        v_new = lr*grad(w[-1]) + gm*v_old 
        w_new = w[-1] - v_new
        if(np.linalg.norm(grad(w_new)) < 1e-6): break 
        w.append(w_new)
        v_old = v_new
    return w[-1], i
        
def SGD(w_init, lr):
    w = [w_init]
    N = Xbar.shape[0] 
    for i in range(100):
        idx = np.random.randint(0, N) 
        xi = Xbar[idx: idx+1] 
        yi = y[idx: idx+1] 
        w_new = w[-1] -lr*grad_sgd(w[-1], xi, yi)
        if(np.linalg.norm(w_new - w[-1]) < 1e-6): break 
        w.append(w_new)
    return w[-1], i 

def MBGD(w_init, lr, batch_size):
    w = [w_init]
    N = Xbar.shape[0] 
    for i in range(100):
        idx = np.random.choice(N, batch_size, replace=False)
        xk = Xbar[idx]
        yk = y[idx]
        w_new = w[-1] - lr*grad_mbgd(w[-1], xk, yk) 
        if(np.linalg.norm(w_new - w[-1]) < 1e-6): break 
        w.append(w_new) 
    return w[-1], i

w_init = np.array([2,2])
w_final, i = GRD(w_init, 0.1)
print(w_final, i)
print(cost(w_final)) # should be close to 0.0

