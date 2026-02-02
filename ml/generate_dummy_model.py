import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# Configuration
MODEL_PATH = "ml/spam_model.h5"
TOKENIZER_PATH = "ml/tokenizer.pkl"
VOCAB_SIZE = 5000
MAX_LENGTH = 100
EMBEDDING_DIM = 64

# 1. Create Dummy Tokenizer
print("Creating dummy tokenizer...")
texts = ["hello world", "spam email"]
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

print(f"Saving tokenizer to {TOKENIZER_PATH}...")
with open(TOKENIZER_PATH, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 2. Build Model (No Training)
print("Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.GlobalAveragePooling1D(), # Simpler than LSTM for dummy
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Save Model
print(f"Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("Done!")
