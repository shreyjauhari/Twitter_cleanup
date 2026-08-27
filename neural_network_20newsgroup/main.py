import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ============================================================
# 1. CHECK DEVICE
# ============================================================

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not detected. Training will run on CPU.")


# ============================================================
# 2. LOAD BBC DATASET
# ============================================================

data_path = r"C:\Users\shaal\Desktop\try\neural_network_20newsgroup\data\bbc-news-data.csv"

# Try reading as tab-separated first
df = pd.read_csv(
    data_path,
    sep="\t",
    on_bad_lines="skip"
)

print("\nDataset loaded successfully!")
print("Columns:", df.columns.tolist())
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CHECK / FIX COLUMN NAMES
# ============================================================

# Rename possible text columns to "text"
if "content" in df.columns and "text" not in df.columns:
    df = df.rename(columns={"content": "text"})

elif "article" in df.columns and "text" not in df.columns:
    df = df.rename(columns={"article": "text"})

# Check required columns
if "text" not in df.columns:
    raise ValueError(
        f"Text column not found. Available columns: {df.columns.tolist()}"
    )

if "category" not in df.columns:
    raise ValueError(
        f"Category column not found. Available columns: {df.columns.tolist()}"
    )

# Remove missing values
df = df.dropna(subset=["text", "category"]).copy()

# Convert to string
df["text"] = df["text"].astype(str)
df["category"] = df["category"].astype(str)

print("\nCleaned dataset shape:", df.shape)

print("\nCategories:")
print(df["category"].value_counts())


# ============================================================
# 4. CONVERT CATEGORY NAMES TO NUMBERS
# ============================================================

categories = sorted(df["category"].unique())

label2id = {
    label: index
    for index, label in enumerate(categories)
}

id2label = {
    index: label
    for index, label in enumerate(categories)
}

df["label"] = df["category"].map(label2id)

print("\nLabel mapping:")
print(label2id)

print("\nNumber of categories:", len(categories))


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining samples:", len(train_texts))
print("Validation samples:", len(val_texts))


# ============================================================
# 6. LOAD ROBERTA TOKENIZER
# ============================================================

MODEL_NAME = "roberta-base"

print(f"\nLoading tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 7. CREATE CUSTOM DATASET CLASS
# ============================================================

class BBCDataset(Dataset):

    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256
        )

        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item

    def __len__(self):
        return len(self.labels)


# Create datasets
print("\nCreating training dataset...")

train_dataset = BBCDataset(
    train_texts,
    train_labels
)

print("Creating validation dataset...")

val_dataset = BBCDataset(
    val_texts,
    val_labels
)


# ============================================================
# 8. LOAD PRETRAINED ROBERTA MODEL
# ============================================================

print(f"\nLoading model: {MODEL_NAME}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(categories),
    id2label=id2label,
    label2id=label2id
)


# ============================================================
# 9. TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(
    output_dir="./results_bbc",

    num_train_epochs=3,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    warmup_ratio=0.1,
    weight_decay=0.01,

    # Use mixed precision on GPU
    fp16=torch.cuda.is_available(),

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    logging_dir="./logs_bbc",
    logging_steps=20,

    disable_tqdm=False,

    dataloader_pin_memory=torch.cuda.is_available(),

    seed=42
)


# ============================================================
# 10. ACCURACY FUNCTION
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = (
        predictions == labels
    ).mean()

    return {
        "accuracy": accuracy
    }


# ============================================================
# 11. CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# ============================================================
# 12. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING ON BBC DATASET")
print("=" * 60)

if torch.cuda.is_available():
    print("Training on GPU:", torch.cuda.get_device_name(0))
else:
    print("Training on CPU")

trainer.train()


# ============================================================
# 13. FINAL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

results = trainer.evaluate()

print("\nFinal Results:")

for key, value in results.items():
    print(f"{key}: {value}")


# ============================================================
# 14. SAVE MODEL AND TOKENIZER
# ============================================================

save_path = "./saved_model_bbc"

os.makedirs(
    save_path,
    exist_ok=True
)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print(f"\nModel saved to: {save_path}")
print(f"Number of categories: {len(categories)}")
print("Categories:", categories)
print("\n" + "=" * 60)
print("TESTING WITH NEW TEXT")
print("=" * 60)


def predict_category(text):

    # Put model in evaluation mode
    model.eval()

    # Tokenize the input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Move input to same device as model
    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Disable gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    # Get predicted label
    predicted_id = torch.argmax(
        probabilities,
        dim=1
    ).item()

    # Get confidence
    confidence = probabilities[0][predicted_id].item()

    # Convert ID back to category
    predicted_category = id2label[predicted_id]

    return predicted_category, confidence


# ============================================================
# TEST WITH CUSTOM TEXT
# ============================================================
print("\n" + "=" * 60)
print("TESTING WITH YOUR OWN TEXT")
print("=" * 60)

while True:
    test_text = input("\nEnter your text (or type 'quit' to exit):\n")

    if test_text.lower() == "quit":
        print("Exiting...")
        break

    if not test_text.strip():
        print("Please enter some text.")
        continue

    category, confidence = predict_category(test_text)

    print("\nPredicted Category:", category)
    print("Confidence:", f"{confidence * 100:.2f}%")