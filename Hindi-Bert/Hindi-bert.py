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
    df_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv')
    df_non_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv')

    df_sarcastic['label'] = 'sarcastic'
    df_non_sarcastic['label'] = 'non_sarcastic'

    df = pd.concat([df_sarcastic, df_non_sarcastic], axis=0)

    columns_to_drop = ['username', 'acctdesc', 'location', 'following', 'followers', 
                       'totaltweets', 'usercreatedts', 'tweetcreatedts', 'retweetcount', 'hashtags']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

    df = df.reset_index(drop=True)

    df['clean_text'] = df['text'].apply(clean_hindi_text)

    duplicates = df[df.duplicated('clean_text', keep=False)]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicates in clean_text.")
        conflict_groups = duplicates.groupby('clean_text')['label'].nunique()
        conflicts = conflict_groups[conflict_groups > 1]
        if not conflicts.empty:
            print(f"Found {len(conflicts)} clean_text entries with conflicting labels.")
            conflict_texts = conflicts.index
            df = df[~df['clean_text'].isin(conflict_texts)]
            print(f"Removed {len(conflict_texts)} conflicting entries.")
        df = df.drop_duplicates(subset='clean_text', keep='first')
        print(f"Removed duplicates, keeping first occurrence. New size: {len(df)}")
    else:
        print("No duplicates found in clean_text.")

    return df

if __name__ == '__main__':
    df = load_data()
    
    print("Total samples:", len(df))
    print("Class distribution:\n", df['label'].value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.3, stratify=df['label'], random_state=91
    )
    
    t = text.Transformer("monsoon-nlp/hindi-bert", maxlen=60, class_names=list(set(y_train.values)))  # Reduced maxlen
    
    trn = t.preprocess_train(X_train.to_numpy(), y_train.to_numpy())
    evalr = t.preprocess_test(X_test.to_numpy(), y_test.to_numpy())
    
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=trn, val_data=evalr, batch_size=8)  # Increased batch_size
    
    history = learner.fit_onecycle(1e-4, 7)  # Increased epochs
    
    def save_training_curves(history):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Val')
        plt.ylim(0.5, 1.0)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
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
    
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    y_pred = predictor.predict(X_test.to_numpy(), return_proba=False)
    if not isinstance(y_pred[0], str):
        y_pred_labels = [t.get_classes()[int(p)] for p in y_pred]
    else:
        y_pred_labels = y_pred
    
    report = classification_report(y_test, y_pred_labels)
    with open('sarcasm_outputs/classification_report.txt', 'w') as f:
        f.write(report)
    
    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'], yticklabels=['Non-Sarcastic', 'Sarcastic'], annot_kws={"size": 22})
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('True', fontsize=20)
    plt.xlabel('Predicted', fontsize=20)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig('sarcasm_outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Results saved in 'sarcasm_outputs' directory.")