import numpy as np
import matplotlib.pyplot as plt
# Tinh dao ham
def grand(x):
    return z

# Tinh gia tri cua ham so
def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(x0, learning_rate):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - learning_rate * grand(x[-1])
        if(abs(grand(x_new)) < 1e-3): # nghiem nho nhat
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(-5, .1)
(x2, it2) = myGD1(5, .1)
print('Solution x1 = %f, cost = %f, after %d iterations' %(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, after %d iterations' %(x2[-1], cost(x2[-1]), it2))

# Ve do thi
x_vals = np.linspace(-10, 10, 1000)  # Gia tri x de ve do thi
cost_vals = cost(x_vals)  # Gia tri cost tuong ung q

plt.figure(figsize=(10, 6))

# Ve do thi cua ham cost
plt.plot(x_vals, cost_vals, label='Cost Function', color='blue')

# # Ve cac diem cost tai moi vong lap cho x1
# cost_x1 = [cost(x) for x in x1]
# plt.scatter(x1, cost_x1, color='red', label='Gradient Descent Path (x1)', zorder=5)

# Ve cac diem cost tai moi vong lap cho x2
cost_x2 = [cost(x) for x in x2]
plt.scatter(x2, cost_x2, color='green', label='Gradient Descent Path (x2)', zorder=5)

# Tieu de va nhan truc
plt.title('Gradient Descent Visualization')
plt.xlabel('x')
plt.ylabel('Cost')
plt.legend()
plt.grid()
plt.show()

# 
def GD_momentum(X0, learning_rate, grad, gamma):
    X = [X0]
    v_old = np.zeros_like(X0) # ban dau van toc ban 0
    for it in range(100):
        v_new = learning_rate*grad(X[-1]) + gamma*v_old
        X_new = X[-1] - v_new
        if np.linalg.norm(X_new) / np.array(X0).size < 1e-3:  # nghiem nho nhat
            break
        X.append(X_new)
        v_old = v_new
    return X
    
# Nesterov accelerated gradient
def GD_NAG(X0, learning_rate, grad, gamma):
    X = [X0]
    v_old = np.zeros_like(X0) # ban dau van toc ban 0
    for it in range(100):
        v_new = learning_rate*grad(X[-1] - gamma*v_old) + gamma*v_old
        X_new = X[-1] - v_new
        if np.linalg.norm(X_new) / np.array(X0).size < 1e-3:  
            break 
        X.append(X_new)
        v_old = v_new
    return X