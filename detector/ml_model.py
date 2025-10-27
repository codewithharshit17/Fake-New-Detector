import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self.model = PassiveAggressiveClassifier(max_iter=50)
    
    def train_model(self, true_csv_path, fake_csv_path):
        # Load both CSV files
        df_true = pd.read_csv('True.csv')
        df_fake = pd.read_csv('Fake.csv')
        
        # Add label columns
        df_true['label'] = 'REAL'
        df_fake['label'] = 'FAKE'
        
        # Combine into one DataFrame
        df = pd.concat([df_true, df_fake], ignore_index=True)
        
        # Change 'text' if your dataset has a different column for the news article
        X = df['text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=7
        )
        
        tfidf_train = self.vectorizer.fit_transform(X_train)
        tfidf_test = self.vectorizer.transform(X_test)
        
        self.model.fit(tfidf_train, y_train)
        
        y_pred = self.model.predict(tfidf_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        if not os.path.exists('media/plots'):
            os.makedirs('media/plots')
        plt.savefig('media/plots/confusion_matrix.png')
        plt.close()
        
        return acc, cm

    def predict(self, text):
        tfidf = self.vectorizer.transform([text])
        prediction = self.model.predict(tfidf)[0]
        proba = self.model.decision_function(tfidf)[0]
        confidence = abs(proba) / (abs(proba) + 1)  # normalize
        return prediction, confidence*100

    def save_model(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        with open(f'{path}vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(f'{path}model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path='models/'):
        with open(f'{path}vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(f'{path}model.pkl', 'rb') as f:
            self.model = pickle.load(f)
