import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("FMLA1Q1Data_train.csv",names=['x1','x2','y'])
print(data.head())
print()

X = np.array(data[['x1','x2']])
Y = np.array(data[['y']])

# Least Squares Solution
num = np.matmul(np.transpose(X),Y)
deno = np.matmul(np.transpose(X),X)
inv = np.linalg.inv(deno)
W = np.matmul(inv,num)
print("W: ",W)
print()

# Computing Weights using stochastic gradient descent Algorithm
# Initializing the weights
b = 0
w = np.random.uniform(low = 50, high = 100, size=(2,)) # Randomly intializing weights sampled from a uniform distribution.
print("Initial Weights w1 and w2: ",w)
print()

No_iterations = 1000
Learning_parameter = 0.01

diff_t = [] # stores ||W-Wml||^2
W_t = [] # stores weights computed for each time step t

for i in range(No_iterations):
    # Randomly sampling 100 datapoints
    samples = np.random.choice(X.shape[0],size=100)
    X_sampled = X[samples]
    Y_sampled = Y[samples]
    
    y_pred = np.dot(X_sampled,w)+b # predicts y based on wi and b
    error = Y_sampled.flatten() - y_pred # computes the error
    
    # computing the gradients
    dw = (-2/100) * np.dot(X_sampled.T,error)
    db = (-2/100) * np.sum(error)
    
    # Updation of parameters
    w = w - Learning_parameter*dw
    b = b - Learning_parameter*db
    
    W_t.append(w)
    
    e = np.linalg.norm(w-W)**2
    diff_t.append(e)
    
    
plt.figure()
plt.title(r'$||\mathbf{W}_t - \mathbf{W}_{\mathrm{ML}}||^2$ vs t, for Learning parameter 0.01')
plt.plot(diff_t)
plt.xlabel('t')
plt.ylabel(r'$||\mathbf{W}_t - \mathbf{W}_{\mathrm{ML}}||^2$')
plt.grid(True)
plt.show()

# to compute the W
W_t_array =  np.array(W_t)
W_T = np.mean(W_t_array,axis=0)
print()
print("Initial difference in weights: ",diff_t[0])
print("Final difference in weights: ",diff_t[-1])
print(f'Weights computed by stochastic gradient descent approach with Learning Parameter: {Learning_parameter} = {W_T}')

