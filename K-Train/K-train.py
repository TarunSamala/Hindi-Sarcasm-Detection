import numpy as np
import pandas as pd
import tensorflow as tf
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Set random seed for reproducibility
tf.random.set_seed(2)

# Create output directory
os.makedirs('sarcasm_outputs', exist_ok=True)

# Clean Hindi text
def clean_hindi_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load and preprocess data
def load_data():
    # Load dataset
    df_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv')
    df_non_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv')

    # Assign labels
    df_sarcastic['label'] = 'sarcastic'
    df_non_sarcastic['label'] = 'non_sarcastic'

    # Combine datasets
    df = pd.concat([df_sarcastic, df_non_sarcastic], axis=0)

    # Drop unnecessary columns
    columns_to_drop = ['username', 'acctdesc', 'location', 'following', 'followers', 
                       'totaltweets', 'usercreatedts', 'tweetcreatedts', 'retweetcount', 'hashtags']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

    # Reset index
    df = df.reset_index(drop=True)

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
    df = load_data()
    
    # Print basic statistics
    print("Total samples:", len(df))
    print("Class distribution:\n", df['label'].value_counts())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.3, stratify=df['label'], random_state=91
    )
    
    # Text Preprocessing with ktrain
    t = text.Transformer("monsoon-nlp/hindi-bert", maxlen=256, class_names=list(set(y_train.values)))
    
    # Preprocess train and test data
    trn = t.preprocess_train(X_train.to_numpy(), y_train.to_numpy())
    evalr = t.preprocess_test(X_test.to_numpy(), y_test.to_numpy())
    
    # Get the model
    model = t.get_classifier()
    
    # Create learner with reduced batch size
    learner = ktrain.get_learner(model, train_data=trn, val_data=evalr, batch_size=4)
    
    # Train the model
    history = learner.fit_onecycle(1e-4, 3)
    
    # Save training curves
    def save_training_curves(history):
        plt.figure(figsize=(10, 4))
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Val')
        plt.ylim(0.5, 1.0)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('sarcasm_outputs/training_curves.png')
        plt.close()
    
    save_training_curves(history)
    
    # Prediction
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    y_pred = predictor.predict(X_test.to_numpy())
    
    # Convert y_pred to labels if necessary
    y_pred_labels = y_pred if isinstance(y_pred[0], str) else [predictor.get_classes()[p] for p in y_pred]
    
    # Classification report
    report = classification_report(y_test, y_pred_labels)
    with open('sarcasm_outputs/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'], yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('sarcasm_outputs/confusion_matrix.png')
    plt.close()
    
    print("Results saved in 'sarcasm_outputs' directory.")