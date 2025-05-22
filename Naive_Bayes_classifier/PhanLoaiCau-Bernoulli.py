from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
import numpy as np

#train data
# hanoi pho chaolong buncha omai banhgio saigon hutiu banhbo
d1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N'])

# test data
d5 = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
d7 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0]])

# train
model = BernoulliNB()
model.fit(train_data, label)

#test
print('Predicting class of d5:', str(model.predict(d5)[0]))
print('Probability of d6 in each class:', model.predict_proba(d6)) 

print(model.predict(d7))