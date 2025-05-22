import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class MF(object):
    # X shape(M, k)
    # W shape(k, N)
    def __init__(self, Y, K, lam = 0.1, Xinit = None, Winit = None, learning_rate = 0.5, max_iter = 1000, print_every = 100):
        self.Y                  = Y     # ultility matrix 
        self.K                  = K     # latent factors dimension
        self.lam                = lam   # regularization parameter
        self.learning_rate      = learning_rate
        self.max_iter           = max_iter # maximum number of iterations
        self.print_every        = print_every # print loss after each a few iter 
        self.n_users            = int(np.max(Y[:, 0])) + 1
        self.n_items            = int(np.max(Y[:, 1])) + 1
        self.n_ratings          = Y.shape[0]
        self.X  = np.random.randn(self.n_items, K) if Xinit is None else Xinit
        self.W  = np.random.randn(K, self.n_users) if Winit is None else Winit
        self.b  = np.random.randn(self.n_items) # bias for items
        self.d  = np.random.randn(self.n_users) # bias for users

    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            # user_id, item_id, rating
            n, m, rating = int(self.Y[i, 0]), int(self.Y[i, 1]), int(self.Y[i, 2])
            L += 0.5*(self.X[m] @ self.W[:, n] + self.b[m] + self.d[n] - rating)**2

        L /= self.n_ratings
        # regularization, dont ever forget this
        return L + 0.5*self.lam*(np.sum(self.X**2)) + np.sum(self.W**2)
    
    def updateXb(self):
        for m in range(self.n_items):
            # get all users who rated item m and get the corresponding ratings
            ids = np.where(self.Y[:, 1] == m)[0] # get the indices of users who rated item m
            user_ids, ratings = self.Y[ids, 0].astype(np.int32), self.Y[ids, 2]
            Wm, dm = self.W[:, user_ids], self.d[user_ids]
            for i in range(30): # 30 iterations for each item
                xm          = self.X[m]
                error       = xm @ Wm + self.b[m] + dm  - ratings
                grad_xm     = error @ Wm.T / self.n_ratings + self.lam*xm 
                grad_bm     = np.sum(error)/ self.n_ratings
                # gradient descent
                self.X[m]   -= self.learning_rate*grad_xm.reshape(-1)
                self.b[m]   -= self.learning_rate*grad_bm
    
    def updateWd(self):
        for n in range(self.n_users):
            # get all items rated by user n, and the corresponding ratings
            ids = np.where(self.Y[:, 0] == n)[0] # get the indices of items rated by user n
            item_ids, ratings = self.Y[ids, 1].astype(np.int32), self.Y[ids, 2]
            Xn, bn = self.X[item_ids], self.b[item_ids]
            for i in range(30):
                wn          = self.W[:, n]
                error       = Xn @ wn + bn + self.d[n] - ratings
                grad_wn     = Xn.T @ error / self.n_ratings + self.lam*wn
                grad_dn     = np.sum(error)/ self.n_ratings
                # gradient descent
                self.W[:, n] -= self.learning_rate*grad_wn.reshape(-1)
                self.d[n]    -= self.learning_rate*grad_dn

    def fit(self):
        for it in range(self.max_iter):
            self.updateXb()
            self.updateWd()
            if(it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y)
                print('iter = %d, loss = %.4f, RMSE train = %.4f' %(it + 1, self.loss(), rmse_train))
    
    def pred(self, u, i):
        """
        predict the rating of user u for item i
        """
        u, i = int(u), int(i)
        pred = self.X[i] @ self.W[:, u] + self.b[i] + self.d[u]
        return max(0, min(5, pred)) # pred should be between 0 and 5 in MoviesLen
    
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0] # number of test
        SE = 0
        for n in range(n_tests):
            pred     =  self.pred(rate_test[n, 0], rate_test[n, 1])
            SE      += (pred - rate_test[n, 2]) ** 2
        
        RMSE = np.sqrt(SE/n_tests)
        return RMSE


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ML/Recommendation_System/ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('ML/Recommendation_System/ml-100k/ua.test', sep='\t', names=r_cols)
rate_train = ratings_base.values
rate_test = ratings_test.values

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs = MF(rate_train, K = 50, lam = .01, print_every = 5, learning_rate = 50, max_iter = 30)
print("Initial Loss:", rs.loss())
rs.fit()

# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print('\nMatrix Factorization CF, RMSE = %.4f' %RMSE)