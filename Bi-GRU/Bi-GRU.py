import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# ---------------------- Configuration ---------------------- #
MAX_LEN = 35               # Maximum sequence length (adjust based on dataset)
VOCAB_SIZE = 12000         # Maximum vocabulary size
EMBEDDING_DIM = 96         # Embedding dimensions
BATCH_SIZE = 128
EPOCHS = 40
OUTPUT_DIR = "bigru_hindi_sarcasm_outputs"
SARCASTIC_PATH = "../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv"
NON_SARCASTIC_PATH = "../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- Enhanced Text Cleaning for Hindi ---------------------- #
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

# ---------------------- Data Loading & Preprocessing ---------------------- #
def load_data(sarcastic_path, non_sarcastic_path):
    sarcastic = pd.read_csv(sarcastic_path)
    non_sarcastic = pd.read_csv(non_sarcastic_path)
    sarcastic['label'] = 1
    non_sarcastic['label'] = 0
    df = pd.concat([sarcastic, non_sarcastic], ignore_index=True)
    df = df.drop_duplicates(subset=['text'])
    df = df.dropna(subset=['text'])
    df['clean_tweet'] = df['text'].apply(clean_text)
    df = df[df['clean_tweet'].str.strip() != '']
    texts = df['clean_tweet'].tolist()
    labels = df['label'].to_numpy()
    return texts, labels

# Load data
texts, labels = load_data(SARCASTIC_PATH, NON_SARCASTIC_PATH)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

# Tokenization and sequence padding
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)  # Fit only on training data
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')

# ---------------------- Bi-GRU Model Definition ---------------------- #
def build_bigru_model():
    inputs = Input(shape=(MAX_LEN,))
    
    # Embedding layer with L2 regularization
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM, embeddings_regularizer=regularizers.l2(1e-4))(inputs)
    
    # SpatialDropout for better generalization
    x = SpatialDropout1D(0.5)(x)
    
    # Bi-GRU layer with L2 regularization and dropout
    x = Bidirectional(GRU(48,
                          activation='tanh',
                          recurrent_activation='sigmoid',
                          reset_after=True,
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_regularizer=regularizers.l2(1e-4),
                          dropout=0.4,
                          recurrent_dropout=0.0))(x)
    
    # Dense layer with regularization and dropout
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.6)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=2e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# ---------------------- Callbacks ---------------------- #
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

# ---------------------- Class Weights ---------------------- #
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# ---------------------- Training ---------------------- #
model = build_bigru_model()
history = model.fit(
    X_train_padded, y_train,
    validation_data=(X_test_padded, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# ---------------------- Save Training Curves ---------------------- #
plt.figure(figsize=(15, 6))

# Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves', pad=10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.5, 1.0)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

# Loss Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves', pad=10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0, 1.5)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- Evaluation & Report Generation ---------------------- #
# Generate predictions
y_pred = (model.predict(X_test_padded) > 0.5).astype(int)

# Save classification report
report = classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic'], zero_division=0)
report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Sarcastic', 'Sarcastic'],
            yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")