import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1,2,3,4,5]]).T
y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
# w = (Xbar.T @ Xbar)^-1 * Xbar.T @ y
w = np.dot(np.linalg.pinv(Xbar.T @ Xbar), Xbar.T @ y)
print(w)
fig, ax = plt.subplots()
img1 = ax.scatter(X,y, cmap="viridis", alpha=0.5); # cmap là cho chọn các màu khác - alpha là chỉnh độ rõ nét của các điểm 
fig.colorbar(img1)

y_pred = Xbar @ w
img2 = ax.plot(X, y_pred)
plt.show()