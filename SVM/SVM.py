import pandas as pd
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuration
SARCASTIC_PATH = "../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv"
NON_SARCASTIC_PATH = "../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv"
OUTPUT_DIR = "svm_hindi_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_hindi_text(text):
    """Enhanced Hindi text cleaning with linguistic preservation"""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#(\w+)', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\u0900-\u097F\s।॰!?,.;:"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_validate_data():
    """Load data with validation and imbalance handling"""
    try:
        sarcastic = pd.read_csv(SARCASTIC_PATH)
        non_sarcastic = pd.read_csv(NON_SARCASTIC_PATH)
        
        for df in [sarcastic, non_sarcastic]:
            if 'text' not in df.columns:
                raise ValueError("Dataset missing required 'text' column")
        
        sarcastic['label'] = 1
        non_sarcastic['label'] = 0
        
        combined = pd.concat([sarcastic, non_sarcastic], ignore_index=True)
        combined['clean_text'] = combined['text'].apply(clean_hindi_text)
        combined = combined[combined['clean_text'].str.len() > 10]
        combined = combined.drop_duplicates(subset=['clean_text'], keep='first')
        
        return combined
        
    except Exception as e:
        print(f"Data loading error: {str(e)}")
        raise

def main():
    # Load and validate data
    df = load_and_validate_data()
    print(f"Dataset size: {len(df)} samples")
    print("Class distribution:\n", df['label'].value_counts(normalize=True))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )
    
    # Create optimized pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=7,
            max_df=0.75,
            sublinear_tf=True
        )),
        ('feature_select', SelectKBest(chi2, k=1500)),
        ('svm', SVC(
            class_weight='balanced',
            probability=True,
            random_state=42,
            kernel='linear'
        ))
    ])
    
    # Parameter grid
    param_grid = {
        'svm__C': [0.1, 0.5, 1, 10]
    }
    
    # Configure grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nTraining model...")
    grid_search.fit(X_train, y_train)
    
    # Save best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'svm_pipeline.pkl'))
    
    # Evaluation
    print("\nBest Parameters:", grid_search.best_params_)
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)  # Define test_accuracy here
    print(f"Train Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    # Generate reports
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'])
    
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write("SVM Classification Report:\n")
        f.write(report)
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title(f'SVM Confusion Matrix\nTest Accuracy: {test_accuracy:.2%}')  # Now using defined variable
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    
    print("\nResults saved to:", os.path.abspath(OUTPUT_DIR))

if __name__ == '__main__':
    main()