"""
Domain-Adaptive Pre-Training (DAPT) Script for NER Models.

This script performs the first stage of a two-stage training process:
1.  **Domain-Adaptive Pre-Training (DAPT):** It takes a pre-trained language
    model (e.g., xlm-roberta-base) and continues its masked language modeling
    (MLM) training on a custom, domain-specific text corpus. This helps the
    model adapt to the specific vocabulary, syntax, and nuances of the target
    domain (e.g., legal or financial texts).
2.  **Fine-Tuning:** The resulting domain-adapted model is then used as a
    starting point for fine-tuning on a specific downstream task, such as
    Named Entity Recognition (NER). (This part is handled in a separate script).

The script is structured as a single procedural flow, handling the entire DAPT
pipeline from data loading to model saving.
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

# ==============================================================================
# --- STAGE 1: DOMAIN-ADAPTIVE PRE-TRAINING (DAPT) ---
# ==============================================================================
print("\n--- STAGE 1: DOMAIN-ADAPTIVE PRE-TRAINING ---")


# --- 1. Load and Prepare the Custom Domain Corpus ---
print("Loading domain corpus for DAPT...")


# Create a search pattern to find all .parquet files within the training data directory.
# This is only for accessing EU Regulatory Corpus
search_path = os.path.join(DOMAIN_CORPUS_DIR, "*.parquet")
all_parquet_files = glob.glob(search_path)
print(f"Found {len(all_parquet_files)} parquet files to load.")


# Loop through each found parquet file path to load its content.
all_texts = []
for file_path in all_parquet_files:
    print(f"Loading text from: {file_path}")
    df = pd.read_parquet(file_path, columns=['text'])  # Read only the 'text' column for the current file
    all_texts.extend(df['text'].tolist())

print(f"\nSuccessfully loaded a total of {len(all_texts):,} documents.")


# Inspect the first few documents to verify the data has loaded correctly.
for i in range(3):
    print(f"\n--- Document {i+1} ---")
    print(all_texts[i][:500] + "...") # Print the first 500 characters for a quick look.


# Convert the Python list of texts into a Hugging Face `Dataset` object,
# which is the required format for the `Trainer`.
print("\nConverting the loaded text list into a Hugging Face Dataset...")
domain_corpus_dict = {"text": all_texts}
domain_corpus = Dataset.from_dict(domain_corpus_dict)


# Free up memory by deleting the large list and dictionary now that the
# Dataset object is created. This is important for very large corpora.
del all_texts
del domain_corpus_dict


# --- 2. Tokenize Data for Masked Language Modeling (MLM) ---
# Load the pre-trained tokenizer corresponding to the base model.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def tokenize_function(examples):
    """Tokenizes a batch of text examples for MLM."""
    return tokenizer(examples["text"], truncation=True, padding=False, return_special_tokens_mask=True)


print("Tokenizing the custom domain corpus...")
# Apply the tokenization function to the entire dataset using `.map()`.
# `batched=True` processes multiple rows at once for speed.
# `num_proc` uses multiple CPU cores to accelerate tokenization.
# `remove_columns` deletes the original 'text' column as it's no longer needed.
tokenized_domain_corpus = domain_corpus.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"]
)


# Initialize the data collator for Masked Language Modeling.
# This component is responsible for creating batches of data during training.
# It will randomly mask 15% of the tokens in each batch, which the model
# must then learn to predict.
data_collator_mlm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


# --- 3. Configure and Set Up the Trainer ---
model_mlm = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(device)


# Define the training arguments for the DAPT phase.
training_args_mlm = TrainingArguments(
    output_dir="./dapt_training_output",  # Directory for checkpoints and logs.
    overwrite_output_dir=True,
    num_train_epochs=1,

    # --- MEMORY-SAVING ARGUMENTS ---
    # These settings are crucial for training large models on consumer-grade GPUs.
    per_device_train_batch_size=2,  # Batch size per GPU. Reduce if you get out-of-memory errors.
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps before updating model weights.
                                    # This simulates a larger batch size (2 * 8 = 16) without using more memory.
    fp16=True,                      # Use 16-bit mixed-precision training to reduce memory usage and speed up training on compatible GPUs.
    # --------------------------------

    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,             # Only keep the last 2 checkpoints to save disk space.
    logging_steps=500,
    prediction_loss_only=True,      # Speeds up logging by only calculating the loss.
)


# Initialize the Trainer with the model, arguments, data collator, and training dataset.
trainer_mlm = Trainer(
    model=model_mlm,
    args=training_args_mlm,
    data_collator=data_collator_mlm,
    train_dataset=tokenized_domain_corpus,
)


# --- 4. Execute Training and Save the Model ---
print("Starting domain-adaptive pre-training on custom corpus...")
# Begin the training process. This will take a significant amount of time.
trainer_mlm.train()


# After training is complete, save the final domain-adapted model and tokenizer.
# The saved model can now be used for the next stage: fine-tuning on the NER task.
print(f"Saving the domain-adapted model to '{DOMAIN_ADAPTED_MODEL_DIR}'")
trainer_mlm.save_model(DOMAIN_ADAPTED_MODEL_DIR)
tokenizer.save_pretrained(DOMAIN_ADAPTED_MODEL_DIR)

print("--- STAGE 1 COMPLETE ---")
