import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Create output directory
os.makedirs('sarcasm_outputs', exist_ok=True)

# Text cleaning function for Hindi tweets
def clean_text(text):
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

# Load and preprocess data
def load_data():
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

# Parameters
VOCAB_SIZE = 20000  # Increased to capture more Hindi vocabulary
MAX_LENGTH = 60     # Increased to capture longer tweets
EMBEDDING_DIM = 256 # Increased for richer representations
BATCH_SIZE = 32
EPOCHS = 4

# Build Bi-LSTM model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        tf.keras.layers.SpatialDropout1D(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            256,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.005),
            recurrent_dropout=0.2
        )),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            128,
            kernel_regularizer=tf.keras.regularizers.l2(0.005),
            recurrent_dropout=0.2
        )),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),  # Adjusted for faster convergence
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_tweet'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']  # Ensure balanced classes in split
    )

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post')
    test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')

    # Calculate class weights
    non_sarcastic_count = sum(df['label'] == 0)
    sarcastic_count = sum(df['label'] == 1)
    total = non_sarcastic_count + sarcastic_count
    class_weights = {
        0: (1 / non_sarcastic_count) * (total / 2.0),
        1: (1 / sarcastic_count) * (total / 2.0)
    }
    print("Class weights:", class_weights)

    # Define callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        min_delta=0.005,
        restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    )

    # Train the model
    model = build_model()
    history = model.fit(
        train_padded,
        y_train,
        epochs=EPOCHS,
        validation_data=(test_padded, y_test),
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, lr_scheduler],
        class_weight=class_weights
    )

    # Save training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history.history['recall'], label='Train')
    plt.plot(history.history['val_recall'], label='Validation')
    plt.title('Recall Curves')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join('sarcasm_outputs', 'training_curves.png'), bbox_inches='tight')
    plt.close()

    # Generate predictions
    y_pred = (model.predict(test_padded) > 0.5).astype(int).flatten()

    # Save classification report
    report = classification_report(y_test, y_pred, target_names=['Non-Sarcastic', 'Sarcastic'], zero_division=0)
    with open(os.path.join('sarcasm_outputs', 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join('sarcasm_outputs', 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    print("All results saved to 'sarcasm_outputs' directory")