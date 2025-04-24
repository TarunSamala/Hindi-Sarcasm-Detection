import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Create output directory
os.makedirs('hindi_sarcasm_outputs', exist_ok=True)

def clean_hindi_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(sarcastic_path, non_sarcastic_path):
    sarcastic = pd.read_csv(sarcastic_path)
    non_sarcastic = pd.read_csv(non_sarcastic_path)
    
    sarcastic['label'] = 1
    non_sarcastic['label'] = 0
    
    data = pd.concat([sarcastic, non_sarcastic], ignore_index=True)
    data['clean_text'] = data['text'].apply(clean_hindi_text)
    return data

VOCAB_SIZE = 15000
MAX_LENGTH = 100
EMBEDDING_DIM = 128
BATCH_SIZE = 256
EPOCHS = 20

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        tf.keras.layers.SpatialDropout1D(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            128,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            recurrent_dropout=0.3
        )),
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

def save_training_curves(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig(os.path.join('hindi_sarcasm_outputs', 'training_curves.png'))
    plt.close()

if __name__ == "__main__":
    # Load data
    data = load_data(
        '../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv',
        '../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv'
    )
    
    # Verify class distribution
    print("\nClass distribution in full dataset:")
    print(data['label'].value_counts())
    
    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        data['clean_text'],
        data['label'],
        test_size=0.2,
        random_state=42,
        stratify=data['label']
    )
    
    # Check training class distribution
    print("\nClass distribution in training set:")
    print(y_train.value_counts())
    
    # Class weights with error handling
    try:
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weights = {i: weight for i, weight in zip(classes, class_weights)}
        print(f"\nClass weights: {class_weights}")
    except Exception as e:
        print(f"\nError calculating class weights: {e}")
        class_weights = None

    # Tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Sequence processing
    train_seq = tokenizer.texts_to_sequences(X_train)
    val_seq = tokenizer.texts_to_sequences(X_val)
    
    train_padded = tf.keras.preprocessing.sequence.pad_sequences(
        train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    val_padded = tf.keras.preprocessing.sequence.pad_sequences(
        val_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2
    )
    
    # Build and train model
    model = build_model()
    history = model.fit(
        train_padded,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_padded, y_val),
        callbacks=[early_stop, lr_scheduler],
        class_weight=class_weights
    )
    
    # Save training curves
    save_training_curves(history)
    
    # Generate reports
    y_pred = (model.predict(val_padded) > 0.5).astype(int).flatten()
    
    # Classification Report
    report = classification_report(y_val, y_pred)
    with open(os.path.join('hindi_sarcasm_outputs', 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join('hindi_sarcasm_outputs', 'confusion_matrix.png'))
    plt.close()

    print("\nAll results saved to 'hindi_sarcasm_outputs' directory")