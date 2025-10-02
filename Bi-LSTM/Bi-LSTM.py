import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D, Attention, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Set random seed for reproducibility
tf.random.set_seed(42)

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

    df_sarcastic['label'] = 1  # Sarcastic
    df_non_sarcastic['label'] = 0  # Non-sarcastic

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

    # Oversample sarcastic class
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df = pd.concat([df_majority, df_minority_upsampled]).reset_index(drop=True)

    return df

# Focal Loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.75):  # Heavily favor minority
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_factor = tf.pow(1.0 - p_t, self.gamma)
        bce = -self.alpha * y_true * tf.math.log(y_pred) - (1 - self.alpha) * (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(focal_factor * bce)

# Configuration
MAX_LEN = 60
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 15

if __name__ == '__main__':
    # Load and preprocess data
    df = load_data()
    
    print("Total samples:", len(df))
    print("Class distribution:\n", df['label'].value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.3, stratify=df['label'], random_state=42
    )
    
    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')
    
    # Class weights
    classes = np.unique(y_train)
    class_weights = {0: 1, 1: 5}  # Heavy weight for sarcastic
    
    # Build Bi-LSTM with Attention
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM, embeddings_regularizer=regularizers.l2(5e-4))(inputs)
    x = SpatialDropout1D(0.5)(x)  # Reduced dropout
    bilstm = Bidirectional(LSTM(32, return_sequences=True, dropout=0.5, recurrent_dropout=0.0))(x)
    attention = Attention()([bilstm, bilstm])
    attention = GlobalAveragePooling1D()(attention)
    x = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(5e-4))(attention)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    loss = FocalLoss(gamma=2.0, alpha=0.75)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.01, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)
    ]
    
    # Train the model
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )
    
    # Save training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
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
    plt.savefig('sarcasm_outputs/Training_curves.png')
    plt.close()
    
    # Predictions with adjusted threshold
    y_prob = model.predict(X_test_seq)
    y_pred = (y_prob > 0.3).astype(int)  # Lower threshold for minority class
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'])
    with open('sarcasm_outputs/Classification_report.txt', 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'], yticklabels=['Non-Sarcastic', 'Sarcastic'], annot_kws={"size": 22})
    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('True', fontsize=20)
    plt.xlabel('Predicted', fontsize=20)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig('sarcasm_outputs/Confusion_matrix.png')
    plt.close()
    
    print("Results saved in 'sarcasm_outputs' directory.")