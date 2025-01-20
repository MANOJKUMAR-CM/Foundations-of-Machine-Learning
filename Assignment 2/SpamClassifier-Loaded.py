import os
import glob
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import warnings

# Suppress overflow warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SpamClassifier:
    def __init__(self, model_file):
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        self.weights = model['weights']
        self.text2vec = model['vectorizer']
        self.stop_words = set(stopwords.words('english'))  # Load stopwords here
    
    def remove_stopwords(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"[@#\$%^&*\(\)\\/\+-_=\[\]\{\}<>]", "", text)
        text = re.sub(r'\d+', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        text = self.remove_stopwords(text)  
        l = WordNetLemmatizer()
        return " ".join(l.lemmatize(word, "v") for word in text.split())
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def Predict(self, X):
        x = self.text2vec.transform([X])
        z = np.dot(x.toarray(), self.weights)
        y = self.sigmoid(z)
        return 1 if y > 0.5 else 0
    
    def predict(self, folder):
        predictions = {}
        for file in glob.glob(os.path.join(folder, "*.txt")):
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                preprocessed_text = self.preprocess(text)
                prediction = self.Predict(preprocessed_text)
                predictions[file] = 1 if prediction == 1 else 0
        return predictions

def Accuracy(Y, Y_pred):
    sum = 0
    for i in range(len(Y)):
        if(Y[i] == Y_pred[i]):
            sum = sum+1
    return sum/ len(Y)

Classifier = SpamClassifier('trained_model.pkl')

results = Classifier.predict('Test')
Results = list(results.values()) # Stores the Model's predictions in the form of a List
df = pd.read_csv("emails.csv") # Load the test data csv file

print(df.head())
print()

l = list(df['spam']) # Store the labels of the emails in List
print("Accuracy on Test Data:", Accuracy(l, Results))

# Test data
confusion_matrix = metrics.confusion_matrix(l, Results)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.tight_layout()
plt.title("Confusion Matrix of Test Data")
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.show()
