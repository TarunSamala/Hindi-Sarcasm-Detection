import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('svm_outputs', exist_ok=True)

# Clean Hindi text
def clean_hindi_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load and clean data from a single CSV file
def load_data(file_path):
    # Load the dataset (assuming columns: 'Tweet', 'label', and possibly 'ID')
    df = pd.read_csv(file_path)
    
    # Rename 'Tweet' to 'text' for consistency (adjust if column name differs)
    if 'Tweet' in df.columns:
        df = df.rename(columns={'Tweet': 'text'})
    
    # Apply text cleaning
    df['clean_text'] = df['text'].apply(clean_hindi_text)
    
    # Check for duplicates in clean_text
    duplicates = df[df.duplicated('clean_text', keep=False)]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicates in clean_text.")
        
        # Check for label conflicts among duplicates
        conflict_groups = duplicates.groupby('clean_text')['label'].nunique()
        conflicts = conflict_groups[conflict_groups > 1]
        
        if not conflicts.empty:
            print(f"Found {len(conflicts)} clean_text entries with conflicting labels.")
            # Remove entries with conflicting labels
            conflict_texts = conflicts.index
            df = df[~df['clean_text'].isin(conflict_texts)]
            print(f"Removed {len(conflict_texts)} conflicting entries.")
        
        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset='clean_text', keep='first')
        print(f"Removed duplicates, keeping first occurrence. New size: {len(df)}")
    else:
        print("No duplicates found in clean_text.")
    
    return df

# Main execution
if __name__ == '__main__':
    # Load and preprocess data
    # Adjust the file path to your actual dataset location
    data = load_data('hindi_tweets.csv')  # Example path, update as needed
    
    # Print basic statistics
    print("Total samples:", len(data))
    print("Class distribution:\n", data['label'].value_counts())
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        data['clean_text'], data['label'], test_size=0.3, stratify=data['label'], random_state=42
    )
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # SVM with hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_tfidf, y_train)
    
    # Best model
    best_svm = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_svm, X_train_tfidf, y_train, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())
    
    # Evaluate on validation set
    y_pred = best_svm.predict(X_val_tfidf)
    accuracy = best_svm.score(X_val_tfidf, y_val)
    print("Validation Accuracy:", accuracy)
    
    # Save classification report
    report = classification_report(y_val, y_pred)
    with open('svm_outputs/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Save confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'], yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('svm_outputs/confusion_matrix.png')
    plt.close()
    
    print("Results saved in 'svm_outputs' directory.")