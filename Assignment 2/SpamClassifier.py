import numpy as np
import pandas as pd
import glob
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress overflow warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

data = []
label = []

for file in glob.glob(os.path.join("Spam","*.txt")):
    l = 'Spam'
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        s = f.read()
        data.append(s)
        label.append(l)

for file in glob.glob(os.path.join("Ham","*.txt")):
    l = 'Ham'
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        s = f.read()
        data.append(s)
        label.append(l)


label_encoding = {'Spam': 1,'Ham': 0}
label_mapped = [label_encoding[i] for i in label] # Mapping "Spam" to 1 and "Non Spam(Ham)" to 0

Data = pd.DataFrame(data, columns=['email'])
print(Data.head()) # Printing the first 5 rows of the dataframe
print()
print('Contents of Sample email before preprocessing:')
print(Data['email'][0]) # Printing sample email before preprocessing
print()

# To reduce all the elements to lower case
Data['email'] = Data['email'].str.lower()

# To remove puntuations
Data['email'] = Data['email'].apply(lambda x: re.sub(r"[^\w\s]", "", x))

# To remove special characters
Data['email'] = Data['email'].apply(lambda x: re.sub(r"[@#\$%^&*\(\)\\/\+-_=\[\]\{\}<>]", "", x))

# To remove Number
Data['email'] = Data['email'].apply(lambda x: re.sub(r'\d+', '', x))

# Remove newline characters (\n)
Data['email'] = Data['email'].apply(lambda x: x.replace('\n', ' '))

# Remove extra spaces 
Data['email'] = Data['email'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# Applying Lemmatization
l = WordNetLemmatizer()
Data['email'] = Data['email'].apply(lambda x: " ".join(l.lemmatize(word, "v") for word in x.split()))

# Stop words set from nltk
stop_words = set(stopwords.words('english'))

# Function to remove stop words 
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

Data['email'] = Data['email'].apply(remove_stopwords)

print('Contents of Sample email after preprocessing:')
print(Data['email'][0]) # Printing sample email after preprocessing
print()

data = list(Data['email'])
text2vec = CountVectorizer(min_df = 0.005) # To be present in a minimum of approx 100 documents to be considered in corpus

X_train, X_test, Y_train, Y_test = train_test_split(data, label_mapped, test_size=0.2, random_state=42, stratify=label_mapped)
# Results in 80% training data, 20% test data

x_train = text2vec.fit_transform(X_train)
x_test = text2vec.transform(X_test)

class LOGisticRegression:
    def __init__(self, l_rate= 0.1, epoch= 1000):
        self.l_rate = l_rate
        self.epoch = epoch
        self.w = None
        self.bias = None
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def fit(self, X, Y):
        X = X.toarray()
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.epoch):
            z = np.dot(X, self.w) + self.bias
            y_pred = self.sigmoid(z)
            
            dw = (1/n_samples) * np.dot(np.transpose(X),(y_pred-Y))
            db = (1/n_samples) * np.sum(y_pred - Y)
            
            self.w -= self.l_rate * dw
            self.bias -= self.l_rate * db
        
        
    
    def predict(self, X):
        z = np.dot(X.toarray(), self.w) 
        y= self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in y]
    
    def getWeights(self):
        return self.w
    

M = LOGisticRegression() # Creating an object of the class
M.fit(x_train, Y_train) # fitting the model

Y = M.predict(x_train)
# Accuracy of Train data
print("Accuracy on Train data: ",sum(x == y for x,y in zip(Y, Y_train))/ len(Y))
print()

# Accuracy on Test data
Y_test_pred = M.predict(x_test)
print("Accuracy on Test data:", sum(x == y for x,y in zip(Y_test_pred, Y_test))/ len(Y_test))
print()

# Train data
confusion_matrix = metrics.confusion_matrix(Y_train, Y)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.tight_layout()
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.show()

# Test data
confusion_matrix = metrics.confusion_matrix(Y_test, Y_test_pred)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.tight_layout()
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.show()

W = M.getWeights()

with open("Weights.txt",'w') as f:
    for w in W:
        f.write(f"{w}\n")


import pickle
with open('trained_model.pkl','wb') as f:
    pickle.dump({'weights':M.getWeights(), 'vectorizer': text2vec}, f)


