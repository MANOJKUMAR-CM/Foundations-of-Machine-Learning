import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("FMLA1Q1Data_train.csv", names=['x1','x2','y'])
print(data.head())
print()

X = np.array(data[['x1','x2']])
Y = np.array(data[['y']])

# Least Squares solution
num = np.matmul(np.transpose(X),Y)
deno = np.matmul(np.transpose(X),X)
inv = np.linalg.inv(deno)
W = np.matmul(inv,num)
print("W: ",W)
print()

# Computing Weights using gradient descent Algorithm
# Initializing weights
b = 0 # bias
w = np.random.uniform(low=50, high=100, size=(2,)) # Randomly intializing weights sampled from a uniform distribution.
print("Initial Weights w1 and w2: ",w)
No_iterations = 1000
Learning_parameter = 0.01
samples =  len(Y)
print("Total no of samples: ",samples)
print()

diff_t = [] # stores ||W-Wml||^2

for i in range(No_iterations):
        y_pred = np.dot(X,w)+b # predicts y based on wi and b
        error = Y.flatten() - y_pred # computes the error

        # Computing the gradients 
        dw = (-2/samples)*np.dot(np.transpose(X),error)
        db = (-2/samples)*np.sum(error)

        # Updation of parameters
        w = w - Learning_parameter*dw
        b = b - Learning_parameter*db

        e = np.linalg.norm(w - W)**2
        diff_t.append(e)

print("Initial difference in weights: ",diff_t[0])
print("Final difference in weights: ",diff_t[-1])
print(f'Weights computed by gradient descent approach with Learning Parameter: {Learning_parameter} = {w}')
print()

# Plotting
plt.figure()
plt.title(r'$||\mathbf{W}_t - \mathbf{W}_{\mathrm{ML}}||^2$ vs t, for Learning parameter 0.01')
plt.plot(diff_t)
plt.xlabel('t')
plt.ylabel(r'$||\mathbf{W}_t - \mathbf{W}_{\mathrm{ML}}||^2$')
plt.grid(True)
plt.show()


print("For the same randomly initialized w and no of iterations, computing the w's for different values of learning parameter.")
print()
b = 0
w = np.random.uniform(low=50, high=100, size=(2,))
w1 = w

Learning_parameter = [0.01,0.001,0.0001]


for L_parameter in Learning_parameter:
    diff_t = [] # stores ||W-Wml||^2
    for i in range(No_iterations):
        y_pred = np.dot(X,w)+b # predicts y based on wi and b
        error = Y.flatten() - y_pred # computes the error

        # Computing the gradients 
        dw = (-2/samples)*np.dot(np.transpose(X),error)
        db = (-2/samples)*np.sum(error)

        # Updation of parameters
        w = w - L_parameter*dw
        b = b - L_parameter*db

        e = np.linalg.norm(w - W)**2
        diff_t.append(e)
    
    plt.figure()
    plt.title(f'Learning Parameter: {L_parameter}')
    plt.plot(diff_t)
    plt.xlabel('t')
    plt.ylabel(r'$||\mathbf{W}_t - \mathbf{W}_{\mathrm{ML}}||^2$')
    plt.grid(True)
    plt.show()
    print(f'Initial difference in weights by gradient descent approach with Learning Parameter {L_parameter} = {diff_t[0]}')
    print(f'Final difference in weights by gradient descent approach with Learning Parameter: {L_parameter} = {min(diff_t)}')
    print(f'Weights computed by gradient descent approach with Learning Parameter: {L_parameter} = {w}')
    print()
    
    w = w1 # intializing w to intial random weights
    
    
    
