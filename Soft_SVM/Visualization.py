import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(42)

means = [[2,2],[4,2]]
cov = [[.7, 0],[0,.7]]

N = 20 
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1)) # shape (40, 2)
y = np.concatenate((np.ones(N), -np.ones(N))) # shape (40,)

fig, ax = plt.subplots(figsize = (8,6))

ax.scatter(X[y == 1][:,0], X[y == 1][:, 1], color='blue', label = 'Class +')
ax.scatter(X[y == -1][:,0], X[y == -1][:,1], color='red', label = 'Class -')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Dữ liệu 2 lớp với Gaussian ')
ax.legend()
ax.grid(True)
ax.set_aspect('equal') # Ti le truc bang nhau 

plt.show()



