

import os
import sys

import torch

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ============================================================
# 1. FIND THE PROJECT ROOT
# ============================================================

# backend/app.py -> twitter_feed/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to your already trained model
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "neural_network_20newsgroup",
    "saved_model_bbc"
)

print("Project root:", PROJECT_ROOT)
print("Model path:", MODEL_PATH)


# ============================================================
# 2. CHECK IF MODEL EXISTS
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model folder not found!")
    print("Expected path:")
    print(MODEL_PATH)
    sys.exit(1)


# ============================================================
# 3. CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Twitter News Category Filter API",
    description="Classifies text into BBC news categories",
    version="1.0.0"
)


# ============================================================
# 4. ENABLE CORS
# ============================================================

# Allows your browser extension to communicate with this API
app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "https://x.com",
        "https://www.x.com",
        "https://twitter.com",
        "https://www.twitter.com",
    ],

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],

    allow_headers=[
        "Content-Type"
    ],
)

# ============================================================
# 5. LOAD MODEL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Loading trained RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


# ============================================================
# 6. SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Model loaded successfully!")


# ============================================================
# 7. INPUT FORMAT
# ============================================================

class TextRequest(BaseModel):
    text: str


# ============================================================
# 8. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Twitter News Filter API is running!",
        "status": "online"
    }


# ============================================================
# 9. PREDICTION ROUTE
# ============================================================

@app.post("/predict")
def predict(request: TextRequest):

    text = request.text.strip()

    # Check for empty input
    if not text:
        return {
            "error": "Text cannot be empty"
        }

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Move tensors to same device as model
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

    # --------------------------------------------------------
    # CONVERT LOGITS TO PROBABILITIES
    # --------------------------------------------------------

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    # Get highest probability
    predicted_id = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_id
    ].item()

    # Get category name
    predicted_category = model.config.id2label[
        predicted_id
    ]
 
    return {
        "text": text,
        "category": predicted_category,
        "confidence": round(confidence * 100, 2)
    }


# ============================================================
# 10. RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )