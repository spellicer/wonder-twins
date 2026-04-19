import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer
from wonder import ModelConfig, load_dataset

# Load the "Bespoke-Stratos-17k" dataset from bespokelabs
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default")

# Access the first sample in the training set
bespoke_rl['train'][0]
# Model and Output Configuration (same as before, or adjust as needed)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-SFT-training" # New output directory for SFT model
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Arguments - similar to GRPO, but adjust for SFT
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,         # Adjust epochs as needed
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,        # Adjust learning rate for SFT
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="no",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    dataloader_num_workers=2,
    seed=42,
    bf16=True,
    push_to_hub=False,
    gradient_checkpointing=True,
    report_to="none",
)

# Model Configuration - same as before
model_args = ModelConfig(MODEL_NAME)
# Load Bespoke-Stratos-17k dataset
dataset_sft = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split='train') # Only using train split for simplicity

# Initialize tokenizer - same as before
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Initialize base model for SFT - same as before
model_sft = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Initialize the SFT Trainer
sft_trainer = SFTTrainer(
    model=model_sft,                     # Our initialized Qwen model
    train_dataset=dataset_sft,           # Bespoke-Stratos-17k dataset
    processing_class=tokenizer,                 # Tokenizer
    args=training_args,                  # Training arguments
)

# Start the SFT Training Loop
sft_train_result = sft_trainer.train()
# Saving the Trained SFT Model
TRAINED_SFT_MODEL_PATH = "data/Qwen-SFT-training" # Same as OUTPUT_DIR

# Save the tokenizer
tokenizer.save_pretrained(TRAINED_SFT_MODEL_PATH)

# Save the trained model
sft_trainer.save_model(TRAINED_SFT_MODEL_PATH)

print(f"SFT Trained model saved to {TRAINED_SFT_MODEL_PATH}")
