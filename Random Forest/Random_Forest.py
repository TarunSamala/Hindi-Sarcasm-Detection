import pandas as pd
import numpy as np
import re
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR = "sarcasm_outputs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Clean Hindi tweet text"""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (keep the text)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Keep Hindi characters (Devanagari script), English letters, spaces, and punctuation
    text = ''.join([char for char in text if (0x0900 <= ord(char) <= 0x097F) or char.isalpha() or char.isspace() or char in '!?.,:;'])
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data():
    """Load and preprocess dataset"""
    sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv')
    non_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv')
    sarcastic['label'] = 1
    non_sarcastic['label'] = 0
    df = pd.concat([sarcastic, non_sarcastic], ignore_index=True)
    df = df.drop_duplicates(subset=['text'])
    df = df.dropna(subset=['text'])
    df['clean_tweet'] = df['text'].apply(clean_text)
    df = df[df['clean_tweet'].str.strip() != '']
    print("Class distribution:")
    print(df['label'].value_counts())
    return df

# Load and split data
df = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_tweet'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, 
                            class_weight='balanced',
                            n_jobs=-1,
                            random_state=42)
rf.fit(X_train_tfidf, y_train)

# Generate predictions
y_pred = rf.predict(X_test_tfidf)

# Save classification report
report = classification_report(y_test, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'], zero_division=0)
with open(os.path.join(OUTPUT_DIR, 'classification_report-rf.txt'), 'w') as f:
    f.write("Random Forest Classification Report:\n")
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Sarcastic', 'Sarcastic'],
            yticklabels=['Non-Sarcastic', 'Sarcastic'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix-rf.png'), bbox_inches='tight', dpi=300)
plt.close()

print("Results saved to:", OUTPUT_DIR)