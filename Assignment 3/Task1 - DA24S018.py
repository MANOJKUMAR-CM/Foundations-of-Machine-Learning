from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

ds = load_dataset("ylecun/mnist")
Class = defaultdict(list)

for i in ds['train']:
    label = i['label']
    
    if(len(Class[label]) < 100):
        Class[label].append(i)
    if all(len(values) == 100 for values in Class.values()):
        break


# To check if all the classes are sampled
l = []
for i in Class.keys():
    l.append(i)
l.sort()
print(l)

# To check the number of samples per class
for i in range(10):
    print(f'The No of samples in Class {i}:', len(Class[i]))
print()
Dataset = [sample['image'] for values in Class.values() for sample in values] # Sample of 1000 images from Mnist
Data = [np.array(i) for i in Dataset] 
Data = np.array(Data).reshape(1000, -1)


class PCA:
    def __init__(self, ncom = 784):
        self.ncom = ncom
        self.data = None
        
    def fit(self, X):
        self.data = X
        x = X - (np.mean(X, axis=0))
        mat = np.cov(x, rowvar=False)
        
        eig_val, eig_vec = np.linalg.eig(mat)
        eig_val = np.real(eig_val)
        eig_vec = np.real(eig_vec)
        
        index = np.argmax(np.abs(eig_vec), axis=0) # To determine the index of the max value(absolute value) in each eigen vector
        sign = np.sign(eig_vec[index, range(784)]) # To determine the sign of the element of the above index computed
        eig_vec = eig_vec*sign[None,:] # sign[None, :] transforms from 1d array to 2d array (broadcasting)
        eig_vec = eig_vec.T
        
        pairs = [(np.abs(eig_val[i]), eig_vec[i,:]) for i in range(len(eig_val))] # Creating pair of eigen values and corresponding eigen vectors
        pairs.sort(key=lambda x: x[0], reverse=True) # Sorting the pair in the decreasing order of eigen values
        eigval_sorted = np.array([x[0] for x in pairs])
        eigvec_sorted = np.array([x[1] for x in pairs])
        
        self.explained_var = eigval_sorted[:self.ncom] / np.sum(eig_val)
        self.cum_explained_var = np.cumsum(self.explained_var)
        
        self.components = eigvec_sorted[:self.ncom, :]
        
        return self
    
    def transform(self, X):
        X1 = X - (np.mean(X, axis=0))
        return X1.dot(self.components.T)
        
    def inv_transform(self, X_transformed):
        return X_transformed.dot(self.components) + np.mean(self.data, axis=0)
        


p = PCA(25)
p.fit(Data)

# To visualize the image of first 10 components
fig, axes = plt.subplots(2, 5)
for i, ax in enumerate(axes.ravel()):
    ax.imshow(p.components[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Pc{i+1}')
    ax.axis("off")
plt.suptitle("Visualization of First 10 Principal components")
plt.tight_layout()
plt.show()

# The variances explained by each of the principal component
for i in range(25):
    print(f'The variance explained by component {i+1}:', round(p.explained_var[i], 4))

print()
print("Cumulative variance explained by first 25 components:")
print(p.cum_explained_var)
print()
dim = [10,100,140, 200, 250, 500]

for d in dim:
    
    print(f"Images reconstructed using {d} dimensional PCA representations:")
    fig, axes = plt.subplots(1, 10,figsize=(12,2))
    
    P = PCA(d) # Applying PCA
    P.fit(Data)
    X = P.transform(Data)
    X_reconstructed = P.inv_transform(X) # Obtaining reconstructed values of the 
    
    for j in range(10): # Displaying the reconstructed image
        axes[j].imshow(X_reconstructed[j * 100 + 1].reshape(28, 28), cmap="gray")
        axes[j].axis("off")
    plt.suptitle(f"Reconstructed Images using {d} dimension PCA representation")    
    plt.show() # To show only current set of images
    print(f"Amount of Variance captured by {d} dimension PCA representation:", P.cum_explained_var[-1])
    print()


#The 140 dimension representation of the Data captures 95% of variance present in the data and to perform downstream tasks such as digit classification it is sufficient to utilizes the images reconstructed using the 140 dim pca representation.
