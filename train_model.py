import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
import os

# === Step 1: Load and Combine Data ===
print("\n1. Loading CSV files...")
df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

print(f"  True.csv shape: {df_true.shape}")
print(f"  Fake.csv shape: {df_fake.shape}")

df_true['label'] = 'REAL'
df_fake['label'] = 'FAKE'

df = pd.concat([df_true, df_fake], ignore_index=True)
print(f"  Combined shape: {df.shape}")
print("  Labels assigned (REAL, FAKE)\n")

# === Step 2: Select Text and Labels ===
col_name = 'text'   # Adjust if your column is named differently!
if col_name not in df.columns:
    print(f"  ERROR: Column '{col_name}' not found. Available columns: {df.columns}")
    exit()

X = df[col_name]
y = df['label']
print("2. Selected columns for training ('text' for content, 'label' for target).\n")

# === Step 3: Split Data ===
print("3. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}\n")

# === Step 4: TF-IDF Vectorization ===
print("4. Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)
print(f"  Feature matrix shape (train): {tfidf_train.shape}")
print(f"  Feature matrix shape (test):  {tfidf_test.shape}\n")

# === Step 5: Train Model ===
print("5. Training PassiveAggressiveClassifier...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)
print("  Model training complete.\n")

# === Step 6: Make Predictions ===
print("6. Predicting on test data...")
y_pred = model.predict(tfidf_test)

# === Step 7: Evaluate ===
print("7. Evaluation metrics on test data:")
accuracy = accuracy_score(y_test, y_pred)
print(f"  Accuracy: {accuracy*100:.2f}%")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred, labels=['REAL','FAKE'])
print("  Confusion Matrix:\n", cm)

# === Step 8: Visualize Confusion Matrix ===
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
if not os.path.exists('media/plots'):
    os.makedirs('media/plots')
plt.savefig('media/plots/confusion_matrix.png')
plt.close()
print("  Confusion matrix plot saved as 'media/plots/confusion_matrix.png'.\n")

# === Step 9: Save Model and Vectorizer ===
print("8. Saving trained model and vectorizer...")
if not os.path.exists('models'):
    os.makedirs('models')
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  Model and vectorizer saved to /models directory.")

print("\n=== Training Script Complete ===")

report = classification_report(y_test, y_pred, output_dict=True)
scores_df = pd.DataFrame(report).transpose().loc[['FAKE','REAL']][['precision','recall','f1-score']]
scores_df.plot(kind='bar', figsize=(6,4), colormap='Accent')
plt.title('Classification Report (Precision, Recall, F1)')
plt.xticks(rotation=0)
plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig('media/plots/classification_report.png')
plt.close()
