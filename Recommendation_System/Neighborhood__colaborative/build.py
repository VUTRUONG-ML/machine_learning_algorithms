import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class uuCF(object):
    def __init__(self, Y_data, k, sim_func = cosine_similarity):
        self.Y_data = Y_data  # 2d array shape(n_users, 3)
        # each row of Y_data has form [user_id, item_id, rating]
        self.k = k            # number of neighborhood 
        self.sim_func = sim_func # similarity function, default : cosine_similrity
        self.Ybar = None        # Normalize dataa
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 # number of users 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1 # number of items 

    def fit(self):
        # nomalize Y_data -> Ybar
        users = self.Y_data[:, 0] 
        self.Ybar = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            # row indices of ratings made by user n
            ids       = np.where(users == n)[0].astype(np.int32)
            # indices of all items rated by user n 
            item_ids  = self.Y_data[ids, 1] # cột 1 là id của item
            # ratings made by user n
            ratings   = self.Y_data[ids, 2] # cột 2 là rating của user cho item 
            # avoid zero division
            self.mu[n] = np.mean(ratings) if ids.size > 0 else 0 
            self.Ybar[ids, 2] = ratings - self.mu[n]

        ## from the rating matrix as a spare matrix
        self.Ybar = sparse.coo_matrix((self.Ybar[:, 2],
                                       (self.Ybar[:, 1], self.Ybar[:, 0])), (self.n_items, self.n_users)).tocsr()
        self.S = self.sim_func(self.Ybar.T, self.Ybar.T)

    def predict(self, u, i):
        "predict the rating of user u for item i"      
        # find item i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # all users who rated i 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # similarity of u and users who rated i 
        sim           = self.S[u, users_rated_i]
        # most k similarity 
        nns           = np.argsort(sim)[-self.k:] # argsort làm hàm sắp xếp tăng dần trả về index, [-self.k:] lấy k phần tử cuối cùng
        nearest_s     = sim[nns]   # and coresponding similarities 
        # the corresponding ratings 
        r             = self.Ybar[i, users_rated_i[nns]]
        eps           = 1e-8    # a small number to avoid zero division
        return (r*nearest_s).sum()/(np.abs(nearest_s).sum() + eps) + self.mu[u]
    
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ML/Recommendation_System/ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('ML/Recommendation_System/ml-100k/ua.test', sep='\t', names=r_cols)
rate_train = ratings_base.values
rate_test = ratings_test.values
# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs = uuCF(rate_train, k = 40)
rs.fit()

n_tests = rate_test.shape[0]

SE = 0 # squared error
for n in range(n_tests):
    pred = rs.predict(rate_test[n, 0], rate_test[n , 1])
    SE += (pred - rate_test[n, 2])**2

RMSE = np.sqrt(SE/n_tests)
print('User-user CF, RMSE =', RMSE)


## i-iCF 
rate_train = rate_train[:, [1, 0, 2]]
rate_test = rate_test[:, [1, 0, 2]]
rs = uuCF(rate_train, k = 40)
rs.fit()    
n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    predi = rs.predict(rate_test[n, 0], rate_test[n, 1])
SE += (predi - rate_test[n, 2])**2
RMSE = np.sqrt(SE/n_tests)
print('Item-item CF, RMSE =', RMSE)