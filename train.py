import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# =====================================================
# HuggingFace Download Settings
# =====================================================

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

MODEL_NAME = "distilbert-base-uncased"

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing Device : {device}")

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("data/train.csv")

print("\nDataset Shape:", df.shape)

df = df[["text", "sentiment"]]

df = df.dropna()

df["text"] = df["text"].astype(str)

# =====================================================
# Label Encoding
# =====================================================

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["labels"] = df["sentiment"].map(label_map)

df = df.dropna(subset=["labels"])

df["labels"] = df["labels"].astype(int)

df = df[["text", "labels"]]

print("\nLabel Distribution\n")
print(df["labels"].value_counts())

# =====================================================
# Reduce Dataset Size (Optional)
# =====================================================

SAMPLE_SIZE = min(5000, len(df))

df = df.sample(
    n=SAMPLE_SIZE,
    random_state=42
)

# =====================================================
# Train Test Split
# =====================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["labels"]
)

print("\nTraining Samples :", len(train_df))
print("Validation Samples:", len(val_df))

# =====================================================
# Convert to HuggingFace Dataset
# =====================================================

train_dataset = Dataset.from_pandas(
    train_df.reset_index(drop=True)
)

val_dataset = Dataset.from_pandas(
    val_df.reset_index(drop=True)
)

# =====================================================
# Load Tokenizer
# =====================================================

try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )
    print("\nLoaded tokenizer from local cache.")

except Exception:

    print("\nDownloading tokenizer...")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_NAME
    )

# =====================================================
# Tokenization
# =====================================================

def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding=True,
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

# =====================================================
# Torch Format
# =====================================================

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

# =====================================================
# Load Model
# =====================================================

try:

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        local_files_only=True
    )

    print("\nLoaded model from local cache.")

except Exception:

    print("\nDownloading model...")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

model.to(device)

# =====================================================
# Data Collator
# =====================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# =====================================================
# Metrics
# =====================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir="./results",

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    learning_rate=2e-5,
    weight_decay=0.01,

    gradient_accumulation_steps=2,

    logging_strategy="steps",
    logging_steps=100,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    report_to="none",

    fp16=torch.cuda.is_available(),

    do_train=True,
    do_eval=True
)
# =====================================================
# Trainer
# =====================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =====================================================
# Train
# =====================================================

print("\nStarting Training...\n")

trainer.train()

# =====================================================
# Evaluate
# =====================================================

print("\nEvaluating Model...\n")

results = trainer.evaluate()

print("\nEvaluation Results\n")

for key, value in results.items():
    print(f"{key} : {value}")

# =====================================================
# Save Model
# =====================================================

trainer.save_model("model")

tokenizer.save_pretrained("model")

print("\nModel saved successfully!")

print("\nSaved inside folder: model/")