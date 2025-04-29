import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
import fasttext
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
MAX_LEN = 64
VOCAB_SIZE = 15000
EMBEDDING_DIM = 300
BATCH_SIZE = 16
EPOCHS = 20

def clean_hindi_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_data():
    sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv')
    non_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv')
    
    sarcastic['label'] = 1
    non_sarcastic['label'] = 0
    
    df = pd.concat([sarcastic, non_sarcastic])
    df['clean_text'] = df['text'].apply(clean_hindi_text)
    
    # Remove duplicates and conflicts
    df = df.drop_duplicates(subset=['clean_text'])
    conflict_texts = df.groupby('clean_text')['label'].filter(lambda x: x.nunique() > 1)
    df = df[~df['clean_text'].isin(conflict_texts)]
    
    return df

def create_embedding_matrix(tokenizer, ft_model):
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    
    for word, i in word_index.items():
        if i >= VOCAB_SIZE:
            continue
        try:
            embedding_matrix[i] = ft_model.get_word_vector(word)
        except:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
    
    return embedding_matrix

def main():
    # Create output directory
    os.makedirs('bilstm_results', exist_ok=True)
    
    # Load and prepare data
    df = load_data()
    print("Class distribution:\n", df['label'].value_counts())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], 
        test_size=0.3, 
        stratify=df['label'], 
        random_state=42
    )
    
    # Tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(X_train)
    
    # Sequence conversion
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    # Padding
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_LEN)
    
    # Load FastText embeddings
    ft_model = fasttext.load_model('cc.hi.300.bin')
    embedding_matrix = create_embedding_matrix(tokenizer, ft_model)
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_LEN,
            trainable=False
        ),
        tf.keras.layers.SpatialDropout1D(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            64, 
            dropout=0.2, 
            recurrent_dropout=0.2,
            return_sequences=False
        )),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Class weights
    class_weights = {0: 1, 1: len(y_train[y_train==0])/len(y_train[y_train==1])}
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    
    # Save training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.savefig('bilstm_results/training_curves.png')
    plt.close()
    
    # Generate reports
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.savefig('bilstm_results/confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    main()