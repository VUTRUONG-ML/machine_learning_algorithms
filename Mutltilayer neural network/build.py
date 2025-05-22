## hàm kích hoạt ReLU
## hàm mất mát là hàm crossetropy
# output là hàm soft max  
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

# Số điểm dữ liệu mỗi lớp
N = 150  # 150 điểm mỗi lớp
D = 2    # Số chiều dữ liệu (2D)
K = 3    # Số lớp

X = np.zeros((N*K, D))  # Ma trận dữ liệu (450, 2)
y = np.zeros(N*K, dtype='int')  # Vector nhãn (450,)

for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)  # Bán kính từ tâm
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # Góc xoắn
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# Trực quan hóa dữ liệu
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Dữ liệu huấn luyện")
plt.show()

# Tính giá trị softmax cho mỗi phần tử trong Z , A2
def softmax_stable(Z):
    """Tinh softmax cho moi tap hop diem trong Z
       each Row of Z is a set of score
    """
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

# Tính hàm mất mát 
def crossentropy_loss(Yhat, y):
    """
    Yhat : shape (Npoint, n class) - predict output
    y : shape(Npoint) -- true label
    NOTE: We don’t need to use the one-hot vector here since most of elements
    are zeros. When programming in numpy, in each row of Yhat, we need to access
    to the corresponding index only.
    """
    id0 = range(Yhat.shape[0])
    # Trong công thức là y thực tế nhân với log của Y dự đoán, mà y thực tế là dạng one hot nghĩa là một vector chỉ có một phần tử bằng 1, nên mình chọn thẳng phần tử bằng một là chỉ số thứ y luôn 
    return -np.mean(np.log(Yhat[id0, y])) # vì y không phải là vector dạng one-hot nó là vector chỉ thẳng point đó thuộc lớp nào luôn, thay vì nhân one-hot điểm nào bẳng 1 thì mới nhân được, thì mình chỉ đích danh của điểm y đó luôn vì class y đó = 1 mới có giá trị ngược lại thì bằng 0 (nhân với 0 thì ko có giá trị )

def mlp_init(d0, d1, d2):
    """
    Initialize W1, b1, W2, b2
    d0: dimension of input data
    d1: number of hidden unit
    d2: number of output unit = number of classes
    """
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros(d1)
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros(d2)
    return (W1, b1, W2, b2)

# Dự đoán xem điểm nào thuộc lớp nào
def mlp_predict(X, W1, b1, W2, b2):
    """
    Suppose that the network has been trained, predict class of new points.
    X: data matrix, each ROW is one data point.
    W1, b1, W2, b2: learned weight matrices and biases
    """
    
    Z1 = X @ W1 + b1 # shape(N, d1)
    A1 = np.maximum(Z1, 0) # shape(N, d1) ReLU
    Z2 = A1 @ W2 + b2 # shape(N, 2)
    return np.argmax(Z2, axis=1) 

def mlp_fit(X, y, W1, b1, W2, b2, eta):
    N = X.shape[0]
    loss_hist = []
    for i in range(20000):
        # feedforward
        Z1 = X @ W1 + b1 # shape(N, d1)
        A1 = np.maximum(Z1, 0) # shape(N, d1) ReLU
        Z2 = A1 @ W2 + b2 # shape(N, 2)
        Yhat = softmax_stable(Z2) # shape(N, 2), A2
        if i % 1000 == 0: # print loss after each 1000 iterations
            loss = crossentropy_loss(Yhat, y)
            print("iter %d, loss: %f" %(i, loss))
            loss_hist.append(loss)

        # back propagation 
        id0 = range(Yhat.shape[0])
        Yhat[id0, y] -= 1   # giống như (A - Y) nhưng Y dự đoán trong thực tế là dạng one-hot, có nghĩa là điểm nào bằng 1 thì nó mới trừ, thì đây trừ thẳng cho 1
        E2 = Yhat / N  # shape(N, d2), này là đạo hàm của hàm mất mát theo Z 
        dW2 = A1.T @ E2 # shape(d1, d2) , đạo hàm của hàm mất mát theo Wij 
        db2 = np.sum(E2, axis=0)         # shape(d2, )
        E1 = E2 @ W2.T                   # shape(N, d1)
        E1[Z1 <= 0] = 0                  # shape(N, d1) , đạo hàm của hàm A1 theo Z1 (đạo hàm ReLU)
        dW1 = X.T @ E1                   # shape(d0, d1)
        db1 = np.sum(E1, axis=0)         # shape(d1, )

        #Gradiant descent update
        W1 += -eta*dW1
        b1 += -eta*db1
        W2 += -eta*dW2
        b2 += -eta*db2
    return (W1, b1, W2, b2, loss_hist)

# suppose X, y are training input and output, respectively
d0 = 2             # data dimension
d1 = h = 100        # number of hidden units
d2 = C = 3              # number of output units
eta = 1   
(W1, b1, W2, b2) = mlp_init(d0, d1, d2)
(W1, b1, W2, b2, loss_hist) = mlp_fit(X, y, W1, b1, W2, b2, eta)
y_pred = mlp_predict(X, W1, b1, W2, b2)
acc = 100*np.mean(y_pred == y)
print('training accuracy: %.2f %%' % acc)