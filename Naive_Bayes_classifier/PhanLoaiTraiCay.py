import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
random.seed(41)
# Function to generate sample data
def generate_sample_data(num_samples):
    data = []
    for _ in range(num_samples):
        # Randomly choose a label
        label = random.choice(['Apple', 'Orange'])
        texture = random.choice([1, 0])

        if label == 'Apple':
            weight = random.randint(150, 180)  # Weight range for Apple
        else:
            weight = random.randint(100, 149)  # Weight range for Orange

        data.append([weight, texture, label])
    return data

# Generate 50 samples
num_samples = 150
data = generate_sample_data(num_samples)

df = pd.DataFrame(data, columns=['Weight', 'Texture', 'Label'])

X = df[['Weight', 'Texture']]
Y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

# print(y_pred)
# print(y_test)

new_data = pd.DataFrame({'Weight': [160], 'Texture': [0]})
prediction = model.predict(new_data)

print("Prediction:", prediction[0])
apple_data = df[df['Label'] == 'Apple']
orange_data = df[df['Label'] == 'Orange']

plt.hist(apple_data['Weight'], alpha=0.5, label='Apple')
plt.hist(orange_data['Weight'], alpha=0.5, label='Orange')
plt.legend(loc='upper right')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Weight Distribution of Apple vs Orange')
plt.show()