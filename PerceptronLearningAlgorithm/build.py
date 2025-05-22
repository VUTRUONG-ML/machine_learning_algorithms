import numpy as np

def predict(w, X):
    ''' Dự đoán X có N hàng d cột 
    '''
    return np.sign(X.dot(w))

def perceptron(X, y, w_init):
    w = w_init
    while True:
        pred = predict(w, X)
        # tim chỉ số hàng nào dự đoán sai
        mis_indexs = np.where(np.equal(pred, y) == False)[0]
        # đối với pocket algorithm thì khi nào num_mis mới bé hơn num_mis cũ thì mới lấy danh sách num mis mới, còn ngược lại thì giữ nguyên danh sách mis 
        num_mis = mis_indexs.shape[0]
        if(num_mis == 0): return w 
        random_id = np.random.choice(mis_indexs, 1)[0] # chọn 1 hàng ngẫu nhiên trong danh sách mis
        w = w + y[random_id] * X[random_id] # y[random_id] * X[random_id] là gradient của hàm mất mát 
    return -1

means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10 
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1*np.ones(N)))

Xbar = np.concatenate((np.ones((2*N, 1)), X), axis= 1)
w_init = np.random.randn(Xbar.shape[1])
w = perceptron(Xbar, y, w_init)
print(w)