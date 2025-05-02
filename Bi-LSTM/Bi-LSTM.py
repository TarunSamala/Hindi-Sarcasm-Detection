import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configuration
MAX_LEN = 50
VOCAB_SIZE = 15000
EMBEDDING_DIM = 96
BATCH_SIZE = 64
EPOCHS = 40
MODEL_PATH = "optimized_bi_lstm.keras"

def load_dataset():
    """Improved data loading with validation"""
    tweets = {}
    with open("../Dataset/Sarcasm_tweets.txt", "r", encoding='utf-8') as f:
        current_id = ""
        for line in f:
            line = line.strip()
            if line.isdigit():
                current_id = line
            elif current_id:
                tweets[current_id] = line

    labels = {}
    with open("../Dataset/Sarcasm_tweet_truth.txt", "r", encoding='utf-8') as f:
        current_id = ""
        for line in f:
            line = line.strip()
            if line.isdigit():
                current_id = line
            elif current_id:
                labels[current_id] = 1 if line == "YES" else 0

    valid_ids = [id for id in tweets if id in labels]
    return [tweets[id] for id in valid_ids], np.array([labels[id] for id in valid_ids])

def clean_text(text):
    """Enhanced text cleaning preserving linguistic features"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'([!?])\1+', r'\1', text)  # Reduce repeated punctuation
    text = re.sub(r'[^\w\s!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def build_optimized_model():
    """Enhanced Bi-LSTM architecture with robust regularization"""
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding with Gaussian Noise
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM, 
                embeddings_regularizer=regularizers.l2(1e-4),
                mask_zero=True)(inputs)
    x = GaussianNoise(0.1)(x)
    
    # First BiLSTM with Spatial Dropout
    x = Bidirectional(LSTM(64, return_sequences=True,
                         kernel_regularizer=regularizers.l2(1e-4),
                         recurrent_regularizer=regularizers.l2(1e-4),
                         dropout=0.3,
                         recurrent_dropout=0.2))(x)
    x = SpatialDropout1D(0.4)(x)
    
    # Second BiLSTM
    x = Bidirectional(LSTM(32,
                         kernel_regularizer=regularizers.l2(1e-4),
                         recurrent_regularizer=regularizers.l2(1e-4),
                         dropout=0.3,
                         recurrent_dropout=0.2))(x)
    
    # Dense Layers with Regularization
    x = Dense(64, activation='relu', 
             kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', 
             kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Optimizer with Weight Decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=2e-4,
        weight_decay=1e-4,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

def plot_training_curves(history):
    """Enhanced visualization with consistent scaling"""
    plt.figure(figsize=(15, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0.5, 1.0)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(0, 1.5)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and preprocess data
    texts, labels = load_dataset()
    texts = [clean_text(text) for text in texts]
    
    print(f"Class distribution: {np.bincount(labels)}")
    print(f"Sarcastic ratio: {np.mean(labels):.2%}")
    
    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        padded, labels, test_size=0.2, 
        stratify=labels, random_state=42,
        shuffle=True
    )
    
    # Class weights with smoothing
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights * 0.9))  # Smooth class weights
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, 
                     min_delta=0.001, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                         patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]
    
    # Build and train model
    model = build_optimized_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluation
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Save classification report
    report = classification_report(y_test, y_pred, 
                                  target_names=['Non-Sarcastic', 'Sarcastic'],
                                  zero_division=0)
    with open('classification_report.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plot_training_curves(history)

if __name__ == "__main__":
    main()