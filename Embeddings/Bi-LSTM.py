import numpy as np
import pandas as pd
import tensorflow as tf
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import fasttext

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Create output directory
os.makedirs('bilstm_outputs', exist_ok=True)

# Clean Hindi text (same as previous)
def clean_hindi_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and preprocess data (same as previous)
def load_data():
    df_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv')
    df_non_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv')

    df_sarcastic['label'] = 1
    df_non_sarcastic['label'] = 0
    df = pd.concat([df_sarcastic, df_non_sarcastic], ignore_index=True)
    
    df['clean_text'] = df['text'].apply(clean_hindi_text)
    
    # Remove duplicates and conflicts
    df = df.drop_duplicates(subset=['clean_text'])
    conflict_texts = df.groupby('clean_text')['label'].filter(lambda x: x.nunique() > 1)
    df = df[~df['clean_text'].isin(conflict_texts)]
    
    return df

# Load FastText embeddings
def load_ft_embeddings(path):
    print("Loading FastText model...")
    ft_model = fasttext.load_model(path)
    print(f"Embedding dimension: {ft_model.get_dimension()}")
    return ft_model

# Create embedding matrix
def create_embedding_matrix(tokenizer, ft_model, embedding_dim=300):
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_matrix[i] = ft_model.get_word_vector(word)
    
    return embedding_matrix, vocab_size

def main():
    # Load and prepare data
    df = load_data()
    print("Class distribution:\n", df['label'].value_counts())
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=0.3,
        stratify=df['label'],
        random_state=42
    )
    
    # Tokenization
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(X_train)
    
    # Sequence conversion
    train_sequences = tokenizer.texts_to_sequences(X_train)
    val_sequences = tokenizer.texts_to_sequences(X_val)
    
    # Padding
    max_length = 100
    X_train_pad = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
    X_val_pad = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Load FastText embeddings
    ft_model = load_ft_embeddings('cc.hi.300.bin')
    embedding_matrix, vocab_size = create_embedding_matrix(tokenizer, ft_model)
    
    # Model parameters
    embedding_dim = 300
    lstm_units = 128
    dropout_rate = 0.5
    recurrent_dropout = 0.2
    l2_lambda = 0.01
    
    # Build model
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False
        ),
        SpatialDropout1D(dropout_rate),
        Bidirectional(LSTM(
            lstm_units, 
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_lambda)
        )),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))
    ])
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)
    
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2)
    ]
    
    # Train model
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=20,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Save training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('bilstm_outputs/training_curves.png')
    plt.close()
    
    # Generate predictions
    y_pred = (model.predict(X_val_pad) > 0.5).astype(int)
    
    # Classification report
    report = classification_report(y_val, y_pred)
    with open('bilstm_outputs/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig('bilstm_outputs/confusion_matrix.png')
    plt.close()
    
    print("Results saved in 'bilstm_outputs' directory")

if __name__ == '__main__':
    main()