import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Configuration
DATA_PATH = "ml/data/spam.csv"
MODEL_PATH = "ml/spam_model.h5"
TOKENIZER_PATH = "ml/tokenizer.pkl"
VOCAB_SIZE = 5000
MAX_LENGTH = 100
EMBEDDING_DIM = 64
EPOCHS = 1

# 1. Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# 2. Load and Preprocess Data
print("Loading data...")
try:
    if not os.path.exists(DATA_PATH):
        # Fallback for running from root
        DATA_PATH = "data/spam.csv" 
        if not os.path.exists(DATA_PATH):
             # Fallback for running from ml/ dir
             DATA_PATH = "data/spam.csv"

    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}")
    exit()

print("Cleaning data...")
data['text'] = data['text'].apply(clean_text)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

texts = data['text'].values
labels = data['label'].values

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 3. Tokenization & Padding
print("Tokenizing...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

# Save Tokenizer
print(f"Saving tokenizer to {TOKENIZER_PATH}...")
with open(TOKENIZER_PATH, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 4. Build LSTM Model
print("Building LSTM model...")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 5. Train Model
print("Training model...")
history = model.fit(
    train_padded, train_labels,
    epochs=EPOCHS,
    validation_data=(test_padded, test_labels),
    verbose=1
)

# 6. Evaluate and Save
loss, accuracy = model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {accuracy:.4f}")

print(f"Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("Done!")
