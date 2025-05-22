import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
np.random.seed(42)
num_customer = 150
data = {
    "Age": [np.random.randint(18,70) for _ in range(num_customer)],
    "Annual Income": [np.random.randint(20000, 150000) for _ in range(num_customer)],
    "Spending score": [np.random.randint(1, 150) for _ in range(num_customer)],
}

customer_df = pd.DataFrame(data)

print(customer_df)
customer_df["Customer Group"] = ""
for row in range(num_customer):
    incom = customer_df["Annual Income"][row]
    spending = customer_df["Spending score"][row]
    if(incom > 120000 and spending > 120):
        customer_df["Customer Group"][row] = "High"
    elif(100000 >= incom > 30000 and 100 >= spending > 30):
        customer_df["Customer Group"][row] = "Medium"
    else:
        customer_df["Customer Group"][row] = "Low"

print(customer_df)
print("ok")

X = customer_df[["Age", "Annual Income", "Spending score"]]
Y = customer_df["Customer Group"]

X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

k = 5
model = KNeighborsClassifier(n_neighbors=k, p = 2)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(f"Accuracy of {k}NN: {accuracy_score(Y_test, Y_pred)}")

accuracys = []
k_val = range(1, 26)
for i in k_val:
    model = KNeighborsClassifier(n_neighbors=i, p = 2)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracys.append(accuracy)

plt.plot(k_val, accuracys, marker="o")
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.title("Accuracys vs K")
plt.show()