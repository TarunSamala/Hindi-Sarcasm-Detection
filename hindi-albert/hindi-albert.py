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
import nlpaug.augmenter.word as naw

# Configure settings
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.makedirs('sarcasm_outputs_reg', exist_ok=True)

# Enhanced text cleaning
def clean_hindi_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# Text augmentation for minority class
hindi_aug = naw.ContextualWordEmbsAug(
    model_path='l3cube-pune/hindi-albert',
    action="substitute",
    device='cpu',
    aug_max=2  # Maximum 2 substitutions per sample
)

def augment_data(df, label, multiplier=3):
    augmented = []
    for text in df[df.label == label]['clean_text']:
        augmented.extend([hindi_aug.augment(text) for _ in range(multiplier)])
    return pd.DataFrame({'clean_text': augmented, 'label': label})

# Enhanced data loading with regularization
def load_and_regularize_data():
    # Load and clean data
    df_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-SARCASTIC.csv')
    df_non_sarcastic = pd.read_csv('../data/Sarcasm_Hindi_Tweets-NON-SARCASTIC.csv')
    
    df_sarcastic['label'] = 1
    df_non_sarcastic['label'] = 0
    df = pd.concat([df_sarcastic, df_non_sarcastic])
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_hindi_text)
    
    # Remove duplicates and conflicts
    df = df.drop_duplicates(subset=['clean_text'])
    conflict_texts = df.groupby('clean_text')['label'].filter(lambda x: x.nunique() > 1)
    df = df[~df['clean_text'].isin(conflict_texts)]
    
    # Augment minority class
    sarcastic_df = df[df.label == 1]
    augmented_df = augment_data(sarcastic_df, 1)
    
    # Combine and balance
    balanced_df = pd.concat([
        df[df.label == 0].sample(n=len(augmented_df)+len(sarcastic_df), random_state=42),
        sarcastic_df,
        augmented_df
    ]).sample(frac=1, random_state=42)
    
    return balanced_df

# Regularized model architecture
def build_regularized_model(class_weights):
    # Initialize transformer with regularization
    t = text.Transformer(
        "l3cube-pune/hindi-albert",
        maxlen=64,  # Reduced sequence length
        class_names=['non_sarcastic', 'sarcastic']
    )
    
    # Get base model
    model = t.get_classifier()
    
    # Add regularization
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_regularizer = tf.keras.regularizers.l2(0.01)
            layer.activity_regularizer = tf.keras.regularizers.l1_l2(0.01, 0.01)
    
    # Add dropout
    model.layers[-2].rate = 0.3  # Dropout before final layer
    
    # Modify learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=3e-5,
        weight_decay=0.01  # Weight decay regularization
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return t, model

def main():
    # Load and regularize data
    df = load_and_regularize_data()
    print("Balanced class distribution:\n", df['label'].value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    # Build model
    class_weights = {0: 1, 1: 1.5}  # Moderate class weighting
    t, model = build_regularized_model(class_weights)
    
    # Preprocess data
    trn = t.preprocess_train(X_train.values, y_train.values)
    val = t.preprocess_test(X_test.values, y_test.values)

    # Configure learner
    learner = ktrain.get_learner(
        model,
        train_data=trn,
        val_data=val,
        batch_size=16,
        use_multiprocessing=False
    )

    # Training with strong regularization
    history = learner.fit_onecycle(
        3e-5,
        epochs=8,
        checkpoint_folder='checkpoints',
        early_stopping=2,
        class_weight=class_weights
    )

    # Evaluation
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    y_pred = predictor.predict(X_test.values)
    
    # Generate reports
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Regularized Confusion Matrix')
    plt.savefig('sarcasm_outputs_reg/confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    main()