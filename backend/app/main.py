from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os
import contextlib

# Configuration
MODEL_PATH = "ml/spam_model.h5"
TOKENIZER_PATH = "ml/tokenizer.pkl"
MAX_LENGTH = 100

# Global variables for model and tokenizer
model = None
tokenizer = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Loading model and tokenizer...")
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
             model = tf.keras.models.load_model(MODEL_PATH)
             print("Model loaded successfully.")
        except Exception as e:
             print(f"Failed to load model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}")

    # Load Tokenizer
    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, 'rb') as handle:
                tokenizer = pickle.load(handle)
            print("Tokenizer loaded successfully.")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
    else:
        print(f"Tokenizer file not found at {TOKENIZER_PATH}")
    
    yield
    print("Shutting down...")

app = FastAPI(title="Spam Classifier API", lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class EmailRequest(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

# Helper Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Routes
@app.get("/")
def read_root():
    return {"message": "Spam Classifier API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_spam(request: EmailRequest):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or Tokenizer not loaded")

    # Preprocess
    cleaned_text = clean_text(request.message)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

    # Predict
    prediction = model.predict(padded)[0][0]
    
    # Interpret
    is_spam = prediction > 0.5
    label = "SPAM" if is_spam else "NOT SPAM"
    
    return {
        "prediction": label,
        "confidence": float(prediction) if is_spam else float(1 - prediction)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
