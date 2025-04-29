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
import fasttext
import gensim
from gensim.models import KeyedVectors
from sklearn.utils.class_weight import compute_class_weight

# Configure memory settings
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

def clean_hindi_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data():
    # [Keep your existing data loading code unchanged]
    # ...
    return df

def create_limited_embeddings(df, ft_model, embedding_file='custom_embeddings.vec'):
    """Create embeddings file with only vocabulary from the dataset"""
    # Collect all unique words in the dataset
    vocab = set()
    for text in df['clean_text']:
        vocab.update(text.split())
    
    print(f"Creating limited embeddings with {len(vocab)} unique words...")
    
    # Create custom embedding file
    with open(embedding_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(vocab)} 300\n")  # FastText uses 300 dimensions
        for word in vocab:
            try:
                vector = ft_model.get_word_vector(word)
                vector_str = ' '.join(map(str, vector))
                f.write(f"{word} {vector_str}\n")
            except:
                continue  # Skip words not in FastText vocabulary
    
    return embedding_file

def main():
    # Load and preprocess data
    df = load_data()
    
    # Balance classes through undersampling
    min_class = df['label'].value_counts().min()
    balanced_df = df.groupby('label').apply(lambda x: x.sample(min_class)).reset_index(drop=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df['clean_text'], 
        balanced_df['label'], 
        test_size=0.3, 
        stratify=balanced_df['label'], 
        random_state=42
    )

    # Load FastText model in read-only mode
    print("Loading FastText model in read-only mode...")
    ft_model = fasttext.load_model('cc.hi.300.bin', label_prefix='__label__')

    # Create limited embeddings
    embedding_file = create_limited_embeddings(balanced_df, ft_model)
    
    # Initialize text processor with limited embeddings
    t = text.TextProcessor(
        embedding=embedding_file,
        maxlen=50,
        class_names=['non_sarcastic', 'sarcastic'],
        ngram_range=1
    )
    
    # Preprocess data
    trn = t.preprocess_train(X_train.values, y_train.values)
    val = t.preprocess_test(X_test.values, y_test.values)

    # Build lightweight model
    model = text.text_regression_model('lstm', t, lstm_units=32, dense_layers=[])[0]
    
    # Train with memory optimization
    learner = ktrain.get_learner(
        model, 
        train_data=trn,
        val_data=val,
        batch_size=8,
        use_multiprocessing=False
    )
    
    print("Starting training...")
    learner.fit_onecycle(1e-4, 5)
    
    # [Keep your existing evaluation code]
    # ...

if __name__ == '__main__':
    main()