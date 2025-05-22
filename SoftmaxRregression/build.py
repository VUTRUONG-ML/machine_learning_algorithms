import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def softmax(Z):
    """
    Z: a numpy array of shape (N, C)
    return a numpy array of shape (N, C)
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axit = 1, keepdims = True)
    return A

def softmax_table(Z):
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = e_Z / e_Z.sum()
    return A

def softmax_loss(X, y, W):
    """
    W: 2d numpy array of shape (d, C),
    each column correspoding to one output node
    X: 2d numpy array of shape (N, d), each row is one data point
    y: 1d numpy array -- label of each row of X
    """
    A = softmax_table(X .dot(W))
    id0 = range(X.shape[0])
    return -np.mean(np.log(A[id0, y])) # log của log của softmaxtable thứ id0 ( trong N điểm dữ liệu), y( vì chỉ có Yy mới bằng 1 tức là điểm dữ liệu id0 thuộc lớp Yy thuộc C lớp, còn các điểm còn lại bằng 0)

def softmax_grad(X, y, W):
    """
    W: 2d numpy array of shape (d, C),
    each column correspoding to one output node
    X: 2d numpy array of shape (N, d), each row is one data point
    y: 1d numpy array -- label of each row of X
    """
    A = softmax_table(X @ W) # N row, C col
    id0 = range(X.shape[0])
    A[id0, y] -= 1 # A - Y, N row C col, vì A - Y mà Y là nhãn 0 hoặc 1, tại vị trí id0, y là 1, còn lại là 0 nên mình trừ 1 
    return X.T @ A / X.shape[0]

def softmax_fit(X, y, W, lr = 0.01, nloop = 100, tol = 1e-5, batch_size = 10):
    W_old = W.copy()
    it = 0
    lost_hist = [softmax_loss(X, y, W)]
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size)) # số lượng batch
    while it < nloop :
        it += 1
        mis_ids = np.random.permutation(N) # trộn các data 
        for i in range(nbatches):
            batch_ids = mis_ids[batch_size*i : min(batch_size*(i +1), N)] # lấy ra các id của batch
            X_bacth, y_batch = X[batch_ids], y[batch_ids] # lấy ra các dữ liệu của batch
            W -= lr*softmax_grad(X_bacth, y_batch, W)
        lost_hist.append(softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old) / W.size < tol: # nếu W không thay đổi thì dừng
            break
        W_old = W.copy()
    return W, lost_hist

def pred(W, X):
    """
    predict output of each columns of X . Class of each x_i is determined by
    location of max probability. Note that classes are indexed from 0.
    """
    return np.argmax(X@W, axis =1) # Hàm argmax trả về chỉ số của giá trị cao nhất, dòng lệnh này có nghĩa là nó sẽ trả về chỉ số (lớp) có xác suất cao nhất 

C, N = 5, 500 # number of classes and number of points per class
means = [[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]]
cov = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)
X4 = np.random.multivariate_normal(means[4], cov, N)
X = np.concatenate((X0, X1, X2, X3, X4), axis = 0) # each row is a datapoint
y = np.asarray([0]*N + [1]*N + [2]*N+ [3]*N + [4]*N) # labels of each row of X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

Xbar_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis = 1) # bias trick
W_init = np.random.randn(Xbar_train.shape[1], C)
W, loss_hist = softmax_fit(Xbar_train, y_train, W_init, batch_size = 20, nloop=100, lr =0.05)

Xbar_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis = 1)
y_pred = pred(W, Xbar_test)
print(W)
print("Accuracy: ", accuracy_score(y_test, y_pred)) # 0.8