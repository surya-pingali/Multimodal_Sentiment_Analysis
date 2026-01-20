# Install required libraries
!pip install -U transformers accelerate datasets bertviz umap-learn seaborn

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset, DatasetDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Load the IMDb dataset
file_path = 'IMDB Dataset.csv'
imdb_data = pd.read_csv(file_path)

# Map sentiment labels to numerical values
imdb_data['label'] = imdb_data['sentiment'].map({'positive': 1, 'negative': 0})

# Reduce the dataset size
train_data = imdb_data.sample(5000, random_state=42)  # 5k samples for training
remaining_data = imdb_data.drop(train_data.index)
test_data = remaining_data.sample(1000, random_state=42)  # 1k samples for testing

print(f"Train size: {train_data.shape}, Test size: {test_data.shape}")

# Split the reduced dataset into training, validation, and testing sets
train, validation = train_test_split(train_data, test_size=0.2, stratify=train_data['label'], random_state=42)

# Create Hugging Face Dataset objects
dataset = DatasetDict({
    "train": Dataset.from_pandas(train, preserve_index=False),
    "test": Dataset.from_pandas(test_data, preserve_index=False),
    "validation": Dataset.from_pandas(validation, preserve_index=False)
})

# Initialize the tokenizer
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['review'], padding=True, truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True, batch_size=None)

# Prepare the labels for the model
label2id = {0: "negative", 1: "positive"}
id2label = {v: k for k, v in label2id.items()}

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=2, id2label=id2label, label2id=label2id
).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="bert_imdb_sentiment",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True  # Mixed precision
)

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
preds_output = trainer.predict(dataset["test"])
print(preds_output.metrics)

# Generate classification report
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = dataset["test"]["label"]
print(classification_report(y_true, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=label2id.values(), yticklabels=label2id.values())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

# Update model config before saving
model.config.label2id = label2id
model.config.id2label = id2label

# Save the model
trainer.save_model("bert-imdb-sentiment-model")

# Load the model for inference
from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

model.config.id2label

# Test with examples
examples = [
    "The movie was fantastic! I loved every part of it.",
    "Absolutely terrible. Waste of time.",
    "The plot was decent, but the acting was subpar.",
    "I would highly recommend this to my friends!"
]
results = classifier(examples)
print(results)

# Zip the model folder
!zip -r bert-imdb-sentiment-model.zip bert-imdb-sentiment-model

# Download the zipped model
from google.colab import files
files.download("bert-imdb-sentiment-model.zip")

trainer.save_model("bert-imdb-sentiment-model")  # Saves the model in the current directory
tokenizer.save_pretrained("bert-imdb-sentiment-model")  # Save tokenizer as well