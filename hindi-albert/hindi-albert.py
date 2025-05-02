import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU execution
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import ktrain
from ktrain import text
import numpy as np

# 1. Hardcoded Dataset with Explicit Batch Alignment
texts = [
    'सचमुच मजाकिया बात है',  # sarcastic
    'क्या हास्यास्पद विचार है',  # sarcastic
    'यह एक सामान्य ट्वीट है',  # non-sarcastic
    'साधारण टिप्पणी बिना व्यंग्य'  # non-sarcastic
]
labels = np.array([1, 1, 0, 0], dtype=np.int32)  # Exactly 4 samples

# 2. Fixed Custom Transformer Class
class CustomTransformer(text.Transformer):
    def preprocess_train(self, texts, y=None, mode='train', verbose=1):
        """Properly override parent method signature"""
        # Process texts using parent's tokenizer
        inputs = self.preprocess(texts)
        
        # Convert to TensorFlow Dataset with exact batch size
        return tf.data.Dataset.from_tensor_slices((
            {k: v for k, v in inputs.items()},
            y
        )).batch(4)  # Explicit batch size of 4

# 3. Model Configuration
def build_model():
    t = CustomTransformer(
        "l3cube-pune/hindi-albert",
        maxlen=8,
        class_names=['0','1']
    )
    
    model = t.get_classifier()
    model.layers[-1].activation = None  # Logits output
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return t, model

def main():
    # 4. Verified Data Preparation
    t, model = build_model()
    train_dataset = t.preprocess_train(texts, labels)
    
    # 5. Direct Batch Validation
    for batch in train_dataset:
        inputs, lbls = batch
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Labels shape: {lbls.shape}")
        break
    
    # 6. Guaranteed Training
    model.fit(train_dataset, epochs=1)
    print("Training succeeded with exact dimensions!")

if __name__ == '__main__':
    main()