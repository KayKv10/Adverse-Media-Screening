"""
Task-Specific Fine-Tuning Script for Named Entity Recognition (NER).

This script performs the second stage of a two-stage training process. It assumes
that a domain-adapted language model has already been created by a preceding
script (e.g., `ner_unstruct_train.py`).

The pipeline for this script is as follows:
1.  **Load the Domain-Adapted Model:** It loads the model and tokenizer that
    were saved after the Domain-Adaptive Pre-Training (DAPT) stage.
2.  **Load and Prepare NER Data:** It loads a standard NER dataset (like CoNLL-2003),
    tokenizes the text, and carefully aligns the token-level labels to handle
    word-piece tokenization.
3.  **Fine-Tune for NER:** It configures and runs the Hugging Face `Trainer` to
    fine-tune the domain-adapted model on the NER task. This involves teaching the
    model to classify tokens into specific entity categories (e.g., PER, ORG, LOC).
4.  **Evaluate and Save:** After training, it evaluates the model's performance on
    the test set and saves the final, fine-tuned NER model to disk.
"""

# --- Core Library Imports ---
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset, Dataset
import evaluate
import numpy as np
import os
import pandas as pd
import glob

# --- 1. Dynamic Path and Project Configuration ---
# This section dynamically sets up all necessary file paths based on the script's
# current location. This makes the project portable and runnable on any machine
# without changing hardcoded paths.

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(script_dir))

# Construct absolute paths for data directories using the project root.
DATA_DIR = os.path.join(PROJECT_DIR, "data", "ner")
DOMAIN_CORPUS_DIR = os.path.join(DATA_DIR, "train")

# Construct absolute paths for model directories using the project root.
MODELS_DIR = os.path.join(PROJECT_DIR, "models", "NER models")
DOMAIN_ADAPTED_MODEL_DIR = os.path.join(MODELS_DIR, "domain_adapter")
FINAL_MODEL_DIR = os.path.join(MODELS_DIR, "final_model")

# Ensure the output directories for the models exist before training begins.
os.makedirs(DOMAIN_ADAPTED_MODEL_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

# --- 2. Model and Environment Setup ---

# Define the base Hugging Face model to be used for domain adaptation.
BASE_MODEL = "xlm-roberta-base"

# Check for an available CUDA-enabled GPU and set the device accordingly for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =============================================================================
# STAGE 2: TASK-SPECIFIC FINE-TUNING (NER)
# =============================================================================
print("\n--- STAGE 2: TASK-SPECIFIC FINE-TUNING FOR NER ---")

# --- 2a. Load NER Data and Label Mappings ---
print("Loading NER dataset (conll2003)...")
ner_dataset = load_dataset("tomaarsen/conll2003", )
# Load the CoNLL-2003 dataset, a standard benchmark for NER.
ner_feature = ner_dataset["train"].features["ner_tags"]
label_names = ner_feature.feature.names
print("NER Labels:", label_names)


# --- 2b. Load the DOMAIN-ADAPTED Model and Tokenizer ---
print(f"Loading domain-adapted model from '{DOMAIN_ADAPTED_MODEL_DIR}'")
# Load the tokenizer that was saved during the DAPT stage. It's crucial to use
# the same tokenizer for consistency.
tokenizer_ner = AutoTokenizer.from_pretrained(DOMAIN_ADAPTED_MODEL_DIR)


# --- 2c. Prepare the Data for Token Classification ---
def tokenize_and_align_labels(examples):
    """
    Tokenizes text and aligns labels for NER.

    Word-piece tokenizers (like RoBERTa's) split single words into multiple
    sub-word tokens (e.g., "running" -> "run", "##ning"). This function ensures
    that the NER labels are correctly assigned to the first sub-word token of
    each original word, while subsequent sub-word tokens are ignored during
    loss calculation by assigning them a label of -100.
    """
    # Tokenize the input words. `is_split_into_words=True` is vital as the
    # dataset is already pre-tokenized into words.
    tokenized_inputs = tokenizer_ner(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    # For each sentence in the batch...
    for i, label in enumerate(examples["ner_tags"]):
        # Get the mapping from sub-word tokens back to original words.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        # For each sub-word token...
        for word_idx in word_ids:
            # If it's a special token (like [CLS] or [SEP]), assign -100.
            if word_idx is None:
                label_ids.append(-100)
            # If it's the first token of a new word, assign the word's true label.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # If it's a subsequent token of the same word, assign -100 to ignore it.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing and aligning labels for NER task...")
# Apply the alignment function to the entire dataset.
tokenized_ner_dataset = ner_dataset.map(tokenize_and_align_labels, batched=True)
# The data collator will dynamically pad sequences to the longest length in each batch.
data_collator_ner = DataCollatorForTokenClassification(tokenizer=tokenizer_ner)


# --- 2d. Set up and Run the Trainer for NER ---
# Create mappings from integer IDs to string labels and vice versa. These are
# required by the model for proper inference and evaluation.
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}


# Load the domain-adapted model, but this time with a token classification head.
# The `AutoModelForTokenClassification` class automatically adds a new, untrained
# linear layer on top of the pre-trained model body for the NER task.
model_ner = AutoModelForTokenClassification.from_pretrained(
    DOMAIN_ADAPTED_MODEL_DIR,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
).to(device)

seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    """
    Computes NER metrics (precision, recall, F1, accuracy) for evaluation.
    This function processes the model's raw predictions and converts them into
    the format expected by the `seqeval` library.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored indices (-100) and convert predictions and labels back to strings.
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute the metrics.
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"], "recall": results["overall_recall"],
        "f1": results["overall_f1"], "accuracy": results["overall_accuracy"],
    }

# Define the training arguments for the NER fine-tuning phase.
training_args_ner = TrainingArguments(
    output_dir="./ner_training_output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True,                      # Use mixed precision
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",          # Evaluate at the end of each epoch.
    save_strategy="epoch",          # Save a checkpoint at the end of each epoch.
    load_best_model_at_end=True,    # Automatically load the best model (based on validation loss) at the end of training.
    push_to_hub=False,
)

# Initialize the Trainer for the NER task.
trainer_ner = Trainer(
    model=model_ner,
    args=training_args_ner,
    train_dataset=tokenized_ner_dataset["train"],
    eval_dataset=tokenized_ner_dataset["validation"],
    tokenizer=tokenizer_ner,
    data_collator=data_collator_ner,
    compute_metrics=compute_metrics,
)

print("Starting NER-specific fine-tuning...")
# Begin the fine-tuning process.
trainer_ner.train()

# --- 2e. Benchmark and Save the Final Model ---
print("\nEvaluating the final model on the held-out test set...")

# We evaluate it one last time on the test set to get our final benchmark scores.
test_results = trainer_ner.evaluate(tokenized_ner_dataset["test"])
print("\n--- Benchmark Results ---")
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")
print("-------------------------\n")

# Save the fully fine-tuned, best-performing NER model and its tokenizer.
# This model is now ready for production inference.
print(f"Saving the final NER model to '{FINAL_MODEL_DIR}'")
# trainer_ner.save_model(FINAL_MODEL_DIR)

print("--- STAGE 2 COMPLETE ---")