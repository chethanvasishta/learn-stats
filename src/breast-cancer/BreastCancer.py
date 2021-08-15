from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

dataset = read_csv('breast-cancer.csv', header=None)
data = dataset.values

# split the data into features and output
X = data[:, :-1].astype(str)
y = data[:,-1].astype(str)
print("Input: ", X.shape)
print("Output: ", y.shape)

print("Sample X: ", X[:5])
print("Sample y: ", y[:5])

ordinal = OrdinalEncoder()
X = ordinal.fit_transform(X)

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

print("Encoded X: ", X[:5])
print("Encoded y: ", y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
model = LogisticRegression()
model.fit(X_train, y_train)

print("Encoded X_test: ", X_test[:5])
print("Encoded y_test: ", y_test[:5])

# predict on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", np.round(accuracy * 100, 2))