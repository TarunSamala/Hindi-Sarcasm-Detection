import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import fasttext
import fasttext.util
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
MAX_LEN = 35               # Sequence length (keep conservative)
VOCAB_SIZE = 8000          # Reduced vocabulary size
EMBEDDING_DIM = 300        # FastText dimension
BATCH_SIZE = 8             # Reduced batch size
EPOCHS = 40
OUTPUT_DIR = "bigru_hindi_sarcasm_outputs"
SARCASTIC_PATH = "../../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv"
NON_SARCASTIC_PATH = "../../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv"
FASTTEXT_MODEL_PATH = "cc.hi.300.bin"

# GPU memory configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- FastText Setup ---------------------- #
if not os.path.exists(FASTTEXT_MODEL_PATH):
    print("Downloading FastText model...")
    fasttext.util.download_model('hi', if_exists='ignore')
    print("Download completed!")

ft = fasttext.load_model(FASTTEXT_MODEL_PATH)

# ---------------------- Text Cleaning ---------------------- #
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = ''.join([char for char in text if (0x0900 <= ord(char) <= 0x097F) or char.isspace() or char in '!?.,:;'])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------- Data Loading ---------------------- #
def load_data(sarcastic_path, non_sarcastic_path):
    sarcastic = pd.read_csv(sarcastic_path)
    non_sarcastic = pd.read_csv(non_sarcastic_path)
    
    # Efficient concatenation
    df = pd.concat([
        sarcastic.assign(label=1),
        non_sarcastic.assign(label=0)
    ], ignore_index=True)
    
    # Memory-efficient cleaning
    df['clean_tweet'] = df['text'].astype('string').apply(clean_text)
    df = df[df['clean_tweet'].str.strip().astype(bool)]
    return df['clean_tweet'].tolist(), df['label'].to_numpy()

# Load and split data
texts, labels = load_data(SARCASTIC_PATH, NON_SARCASTIC_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, 
    test_size=0.2, 
    stratify=labels, 
    random_state=42
)

# ---------------------- Embedding Matrix ---------------------- #
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Build embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE:
        embedding_matrix[i] = ft.get_word_vector(word)

# ---------------------- Memory-Efficient Model ---------------------- #
def build_bigru_model():
    inputs = Input(shape=(MAX_LEN,))
    
    x = Embedding(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
        mask_zero=True
    )(inputs)
    
    x = SpatialDropout1D(0.4)(x)  # Reduced from 0.5
    
    x = Bidirectional(GRU(32,  # Reduced units
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        kernel_regularizer=regularizers.l2(1e-4),
                        return_sequences=False,
                        dropout=0.3))(x)  # Reduced dropout
    
    x = Dense(32, activation='relu')(x)  # Smaller dense layer
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ---------------------- Training Setup ---------------------- #
# Sequence conversion
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

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

# ---------------------- Evaluation ---------------------- #
y_pred = (model.predict(X_test_padded, batch_size=BATCH_SIZE) > 0.5).astype(int)

# Classification report
report = classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic'], zero_division=0)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Sarcastic', 'Sarcastic'],
            yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.title('Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), bbox_inches='tight')

print(f"\nResults saved to {OUTPUT_DIR}")