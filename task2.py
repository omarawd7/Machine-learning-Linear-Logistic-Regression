import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(r'customer_data.csv')


print(df)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, learnningRate=0.001, nIt=1000):
        self.learnningRate = learnningRate
        self.nIt = nIt
        self.ceta = None
        self.b = None

    def train(self, X, y):
        noOfRaws, n_features = X.shape
        self.ceta = np.zeros(n_features)
        self.b = 0

        for _ in range(self.nIt):
            linear_pred = np.dot(X, self.ceta) + self.b
            predictions = sigmoid(linear_pred)

            dceta = (1/noOfRaws) * np.dot(X.T, (predictions - y))
            db = (1/noOfRaws) * np.sum(predictions-y)

            self.ceta = self.ceta - self.learnningRate*dceta
            self.b = self.b - self.learnningRate*db


    def predict(self, X):
        linear_pred = np.dot(X, self.ceta) + self.b
        y_pred = sigmoid(linear_pred)
        res = [0 if y<=0.5 else 1 for y in y_pred]
        return res

df = df.sample(frac=1).reset_index(drop=True)
print(df)

df_train=df.sample(frac=0.8, random_state=25)
df_test= df.drop(df_train.index)
print(df_train)
print(df_test)

X_train = df_train.iloc[:, 0:-1]
y_train = df_train.iloc[:, -1]
X_test= df_test.iloc[:, 0:-1]
y_test = df_test.iloc[:, -1]

X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())
X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())

print(X_train)

print(y_test)

clf = LogisticRegression(learnningRate=0.7)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc*100)