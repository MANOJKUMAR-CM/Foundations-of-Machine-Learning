import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

data = pd.read_csv("FMLA1Q1Data_train.csv",names=['x1','x2','y'])
print(data.head())

X = np.array(data[['x1','x2']])
Y = np.array(data[['y']])

# Least Squares Solution
num = np.matmul(np.transpose(X),Y)
deno = np.matmul(np.transpose(X),X)
inv = np.linalg.inv(deno)
W = np.matmul(inv,num)
print("W: ",W)
print()

# To perform cross validation
K = KFold(n_splits=5, shuffle=True, random_state=42)

# Searching for the optimal value of lambda
# Initializing weights
b = 0 #bias
w = np.random.uniform(low = 50, high = 100, size=(2,))
print("Initial Weights w1 and w2: ",w)
print()

No_iterations = 1000
Learning_parameter = 0.01 # Based on previous tasks, 0.01 values helps in reaching optimal values at a faster rate.
reg = np.logspace(-4,4,10)
Avg_val = []

for L in reg:
    val = []
    for train_index, val_index in K.split(X):
        X_train, X_validate = X[train_index], X[val_index]
        Y_train, Y_validate = Y[train_index], Y[val_index]
        samples = len(Y_train)
        w = w.copy() # for each fold making sure that the w is the same randomly initialized value
        b = 0

        for i in range(No_iterations):
            y_pred_train = np.dot(X_train,w)+b # predicts value based on wi and b
            error = Y_train.flatten()-y_pred_train  # computes the error

            # computing the gradient
            dw = (-2/samples)*np.dot(X_train.T,error) + 2*L*w
            db = (-2/samples) * np.sum(error)
        
            # updating the weights
            w = w - Learning_parameter*dw
            b = b - Learning_parameter*db

        y_pred_val = np.dot(X_validate,w)+b
        error = Y_validate.flatten() - y_pred_val
        val_error = np.mean(error**2)
        val.append(val_error)
        
    Avg_val.append(np.mean(val))

i = Avg_val.index(min(Avg_val)) 
lambda_min = reg[i]
print("Lambda value which gives minimum validation error: ",lambda_min)
print()

plt.figure()
plt.plot(reg,Avg_val)
plt.grid(True)
plt.xlabel('lambda')
plt.ylabel('Average validation error')
plt.show()

# Obtaining Wr with respect to optimal lambda 

# Initializing weights
b = 0 #bias
w = np.random.uniform(low = 50, high = 100, size=(2,))
print("Initial Weights w1 and w2: ",w)
print()

for i in range(No_iterations):
    y_pred = np.dot(X,w)+b # predicts value based on wi and b
    error = Y.flatten()-y_pred  # computes the error

    # computing the gradient
    dw = (-2/samples)*np.dot(X.T,error) + 2 *lambda_min * w
    db = (-2/samples) * np.sum(error)

    # updating the weights
    w = w - Learning_parameter*dw
    b = b - Learning_parameter*db

print("Final Weights w1 and w2: ",w)
print()
# Reading the Test data

data_test = pd.read_csv("FMLA1Q1Data_test.csv",names=['x1','x2','y'])
print(data_test.head())

X_test = np.array(data_test[['x1','x2']])
Y_test = np.array(data_test[['y']])

# Predicting using the least squares solution
Y_pred = np.dot(X_test,W)

Error_least_squares = np.mean((Y_test - Y_pred)**2)
print("Error using Least squares solution:",Error_least_squares)
print()

# Prediction using Wr computed by Ridge regression
Y_pred_ridge = np.dot(X_test,w)+b
Error_ridge = np.mean((Y_test - Y_pred_ridge)**2)
print("Error using Ridge regression:",Error_ridge)
print()

