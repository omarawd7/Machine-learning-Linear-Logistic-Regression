
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(r'car_data.csv')


print(df)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

class LinearRegression:

    def __init__(self, learningRate = 0.001, noIt=1000):
        self.learningRate = learningRate
        self.noIt = noIt
        self.ceta = None
        self.b = None
        
   

    def fit(self, X, y):
        noOfRaws, noFeatures = X.shape
        self.ceta = np.zeros(noFeatures)
        self.b = 0
    
        arrmse=[]
        arrit=[]

        for i in range(self.noIt):
            y_pred = np.dot(X, self.ceta) + self.b
           
            dceta = (1/noOfRaws) * np.dot(X.T, (y_pred-y))
            db = (1/noOfRaws) * np.sum(y_pred-y)
            self.ceta = self.ceta - self.learningRate * dceta
            self.b = self.b - self.learningRate * db
            z=mse(y,y_pred)
            print(z)
            arrmse.append(z)
            arrit.append(i)
        plt.plot(arrit,arrmse)
        plt.show()
            

    def predict(self, X):
        res = np.dot(X, self.ceta) + self.b
        return res

df = df.sample(frac=1).reset_index(drop=True)
print(df)

ax1 = df.plot.scatter(x='symboling',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='carlength',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='carwidth',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='wheelbase',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='carheight',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='curbweight',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='cylindernumber',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='enginesize',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='boreratio',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='stroke',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='compressionratio',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='horsepower',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='peakrpm',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='citympg',
                     y='price',
                      c='DarkBlue')
ax1 = df.plot.scatter(x='highwaympg',
                     y='price',
                      c='DarkBlue')

df = df[['citympg','highwaympg','enginesize','curbweight','price']]
print(df)

df_train=df.sample(frac=0.8, random_state=25)
df_test= df.drop(df_train.index)

X_train = df_train.iloc[:, 0:-1]
y_train = df_train.iloc[:, -1]
X_test= df_test.iloc[:, 0:-1]
y_test = df_test.iloc[:, -1]
print(X_train)
print(y_train)

X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())
X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())

reg = LinearRegression(learningRate=0.015)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test, predictions)
print("mean square ="+str(mse))