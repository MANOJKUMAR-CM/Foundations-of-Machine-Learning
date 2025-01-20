import numpy as np
import pandas as pd

data = pd.read_csv("FMLA1Q1Data_train.csv")
print(data.head())
print()
# The columns of the csv file is to be named
data = pd.read_csv("FMLA1Q1Data_train.csv", names=['x1','x2','y'])
print(data.head())
print()

X =np.array(data[['x1','x2']])
print("X.Shape: ",X.shape)

Y = np.array(data[['y']])
print("Y.shape: ",Y.shape)
print()

# computing least squares solution
x1 = np.transpose(X)
num = np.dot(x1,Y)
x2 = np.dot(x1,X)
deno = np.linalg.inv(x2)
W = np.matmul(deno,num)

print("least squares solution W:",W)
print()
print("Shape of W:",W.shape)

# Predicting the Y values using the Least squares solution and computing the sum of squared errors.
y_pred = np.matmul(X,W)
Error = np.mean((Y-y_pred)**2)
print("Error:",Error)
