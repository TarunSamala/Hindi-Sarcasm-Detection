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
import pandas as pd

# ---------------------- Configuration ---------------------- #
MAX_LEN = 35
VOCAB_SIZE = 12000
EMBEDDING_DIM = 96
BATCH_SIZE = 64  # Reduced batch size
EPOCHS = 40
OUTPUT_DIR = "bilstm_hindi_sarcasm_outputs"
SARCASTIC_PATH = "../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv"
NON_SARCASTIC_PATH = "../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- Text Cleaning ---------------------- #
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = ''.join([char for char in text if (0x0900 <= ord(char) <= 0x097F) or char.isalpha() or char.isspace() or char in '!?.,:;'])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------- Data Loading ---------------------- #
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
    return df['clean_tweet'].tolist(), df['label'].to_numpy()

texts, labels = load_data(SARCASTIC_PATH, NON_SARCASTIC_PATH)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

# ---------------------- Tokenization ---------------------- #
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')

# ---------------------- Enhanced Model ---------------------- #
def build_bilstm_model():
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

# ---------------------- Training Setup ---------------------- #
# Class weights with smoothing
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights * 0.9))  # Smooth class weights

# Callbacks with validation loss monitoring
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, 
                 min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=4, min_lr=1e-6, verbose=1)
]

# ---------------------- Training ---------------------- #
model = build_bilstm_model()
history = model.fit(
    X_train_padded, y_train,
    validation_data=(X_test_padded, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)


# ---------------------- Visualization ---------------------- #
plt.figure(figsize=(15, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves', pad=10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0.5, 1.0)
plt.grid(linestyle='--', alpha=0.6)
plt.legend()

# Loss plot
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

# ---------------------- Evaluation ---------------------- #
y_pred = (model.predict(X_test_padded) > 0.5).astype(int)

# Classification report
report = classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic'], zero_division=0)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# Confusion matrix
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