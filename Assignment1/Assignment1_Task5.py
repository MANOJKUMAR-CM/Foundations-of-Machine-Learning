import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("FMLA1Q1Data_train.csv", names=['x1','x2','y'])
print("Training data(First 5 samples):" )
print(data.head())
print()
# Plotting the pairplot to see if there exsists a relation between features and target/label

sns.pairplot(data)
plt.show()

print("The pair plot depicts a Non linear relationship between the features and the labels.")

X_train = np.array(data[['x1','x2']])
Y_train = np.array(data[['y']])

data_test = pd.read_csv("FMLA1Q1Data_test.csv", names=['x1','x2','y'])
print("Testing data (First 5 samples):")
print(data_test.head())
print()

X_test = np.array(data_test[['x1','x2']])
Y_test = np.array(data_test[['y']])

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 /(2*sigma**2))

def Kernel_matrix(X, sigma):
    n = X.shape[0]
    K = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            K[i,j] = gaussian_kernel(X[i], X[j], sigma)
    return K

def compute_alpha(X_train, Y_train, sigma):
    K = Kernel_matrix(X_train,sigma)
    n = X_train.shape[0]
    a = np.linalg.inv(K + 100*np.eye(n)).dot(Y_train)
    return a

def predict(X_train, X_test, sigma, alpha):
    n = X_test.shape[0]
    y_pred = np.zeros(n)
    
    for i in range(X_test.shape[0]):
        kernels_sum = 0
        for j in range(X_train.shape[0]):
            kernels_sum += alpha[j]* gaussian_kernel(X_train[j],  X_test[i],sigma)
        y_pred[i] = kernels_sum
    return y_pred

sigma = 1.2
alpha = compute_alpha(X_train, Y_train, sigma)
y_pred = predict(X_train, X_test, sigma, alpha)

# Computing the SSE Error
def SSE(y_pred,y_test):
    return np.mean((y_test - y_pred)**2)

print("SSE by Gaussian Kernel regression: ", SSE(y_pred, Y_test))

deno = np.matmul(np.transpose(X_train),X_train)
num = np.matmul(np.transpose(X_train),Y_train)
W = np.matmul(np.linalg.inv(deno), num)

y_pred_LS = np.matmul(X_test,W)
E = SSE(y_pred_LS, Y_test)

print("SSE by Standard Least Squares Regression: ", E)


