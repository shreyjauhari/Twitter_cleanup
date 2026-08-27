import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path where your trained model was saved
MODEL_PATH = "./saved_model_bbc"

# Load the ALREADY TRAINED model
print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# Put model on GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)


def predict_category(text):

    # Convert text into tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Move input to same device as model
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Prediction only — NO TRAINING
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert output scores to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    # Find highest probability
    predicted_id = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[0][predicted_id].item()

    # Convert number back to category name
    category = model.config.id2label[
        predicted_id
    ]

    return category, confidence


# Keep accepting your inputs
while True:

    text = input(
        "\nEnter news text (type 'quit' to exit):\n> "
    )

    if text.lower() == "quit":
        break

    category, confidence = predict_category(text)

    print("\nPredicted Category:", category)
    print("Confidence:", f"{confidence * 100:.2f}%")