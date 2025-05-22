import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn import linear_model

# Hàm tìm những bộ phim mà người dùng đó đã đánh giá và giá trị của các rating
def get_item_rated_by_user(rate_matrix, user_id):
    """
    rate_matrix là kiểu biểu diễn user_id - item_id - rating
    thì user_id có thể đánh gía nhiều item_id thì sẽ có nhiều user_id trùng nhau 
    """
    y = rate_matrix[:, 0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1
    # but id in python starts from 0
    ids = np.where(y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] - 1 #index start from 0
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)
# Read user file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ML/Recommendation_System/ml-100k/u.user', sep='|', names=u_cols)
n_users = users.shape[0]
print('Number of user:', n_users)
# Read rating file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ML/Recommendation_System/ml-100k/ua.base',sep='\t', names = r_cols)
rating_test = pd.read_csv('ML/Recommendation_System/ml-100k/ua.test',sep='\t', names = r_cols)

rate_train = ratings_base.values
rate_test = rating_test.values

print('number of train rates:', rate_train.shape[0])
print('number of test rates:', rate_test.shape[0])

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL','unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ML/Recommendation_System/ml-100k/u.item',sep='|', names = i_cols, encoding='latin-1')

X0 = items.values
X_train_counts = X0[:, -19:]
print('number of item:', items.shape[0])

print(rate_train[:4, :])

# feature vector for each item
transformer = TfidfTransformer(smooth_idf=True, norm='l2')
X = transformer.fit_transform(X_train_counts.tolist()).toarray()

d = X.shape[1] 
W = np.zeros((d, n_users))
b = np.zeros(n_users)
for n in range(n_users):
    ids, scores = get_item_rated_by_user(rate_train, n) # get item indices and scores for user n
    model = Ridge(alpha=0.01, fit_intercept=True) # Ridge regression
    Xhat = X[ids, :] # get feature vector for each item rated by user n
    model.fit(Xhat, scores) # fit model to get weights and bias wirh x = Xhar, y = scores
    W[:, n] = model.coef_
    b[n] = model.intercept_

Yhat = X @ W + b 
n = 20
np.set_printoptions(precision=2) # 2 digits after .
ids, scores = get_item_rated_by_user(rate_test, n)
print('Rated movies ids :', ids)
print('True ratings :', scores)
print('Predicted ratings:', Yhat[ids, n])

# Tóm lại là: dựa vào feature của item và rating của user đã cho item để dự đoán các rating cho item mà user chưa rating
# Tìm W và b bằng Ridge regression thông qua dữ liệu đã rating rồi từ đó dự đoán rating cho các item khác