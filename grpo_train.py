# Import necessary libraries
import logging
import os
from dataclasses import dataclass, field
# Import PyTorch and Hugging Face Transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Import dataset utilities
from datasets import load_dataset

# Import libraries from TRL (Transformers Reinforcement Learning)
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

from wonder import TRAINED_MODEL_PATH, GRPOScriptArguments, ModelConfig, device, get_callbacks, get_reward_functions, load_math_dataset, test_model_inference, validate_dataset

# Define the path to your trained model (same as OUTPUT_DIR)
#MODEL_NAME = "deepseek-ai/DeepSeek-V3"  # Base model name (can be a local path or Hugging Face Hub model ID)
#MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"# Base model name (can be a local path or Hugging Face Hub model ID)
#MODEL_NAME = "Qwen/Qwen3-0.6B"# Base model name (can be a local path or Hugging Face Hub model ID)
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
if __name__ == '__main__':

    # Load the "AI-MO/NuminaMath-TIR" dataset from DigitalLearningGmbH
    MATH_le = load_dataset("AI-MO/NuminaMath-TIR", "default")  

    # Access the first sample in the training set
    MATH_le['train'][0]
    # Load the "Bespoke-Stratos-17k" dataset from bespokelabs
    bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default") 

    # Access the first sample in the training set
    bespoke_rl['train'][0]

    # Create output directory if it doesn't exist
    os.makedirs(TRAINED_MODEL_PATH, exist_ok=True)

    # Initialize tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16,  # Use float16 for faster training and reduced memory usage (if supported by the hardware)
    )

    print(f"Model parameters: {model.num_parameters():,}")

    # Test the model
    test_input = "how are you?"
    response = test_model_inference(model, tokenizer, device, test_input)
    print(f"Test Input: {test_input}")
    print(f"Model Response: {response}")
    # Load our training dataset and printing train/test size
    dataset = load_math_dataset()

    print(f"Train set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")
    # Validate dataset
    validate_dataset(dataset)
    # Define TrainingArguments from transformers
    training_args = TrainingArguments(
        output_dir=TRAINED_MODEL_PATH,         # Output directory for checkpoints and logs
        num_train_epochs=1,            # Total number of training epochs
        per_device_train_batch_size=8, # Batch size per device during training
        per_device_eval_batch_size=16, # Batch size for evaluation
        gradient_accumulation_steps=2, # Accumulate gradients to simulate larger batch size
        learning_rate=5e-5,            # Initial learning rate for AdamW optimizer
        warmup_steps=0.1,              # Linear warmup over warmup_ratio fraction of training steps
        weight_decay=0.01,             # Apply weight decay to all layers except bias and LayerNorm weights
        logging_steps=10,              # Log every X updates steps
        eval_strategy="steps",         # Evaluate every `eval_steps`
        eval_steps=50,                 # Evaluation and logging steps
        save_strategy="steps",         # Save checkpoint every `save_steps`
        save_steps=50,                 # Save checkpoint every X updates steps
        save_total_limit=2,            # Limit the total amount of checkpoints. Deletes the older checkpoints.
        dataloader_num_workers=2,      # Number of subprocesses to use for data loading
        seed=42,                       # Random seed for reproducibility
        push_to_hub=False,             # Whether to push the final model to Hugging Face Hub
        gradient_checkpointing=True,   # Enable gradient checkpointing
        report_to="none",              # Reporting to no one
        remove_unused_columns=False,   # Do not remove unused columns from the dataset
        bf16=False,                      # Use mixed precision (if supported by the hardware
        use_cpu=True,                    # Use GPU if available
    )
    # Instantiate configuration objects
    script_args = GRPOScriptArguments()
    model_args = ModelConfig(MODEL_NAME)
    logger = logging.getLogger(__name__)
    logging.basicConfig() # Ensure at least one handler exists
    logging.getLogger().setLevel(logging.INFO)

    # Get reward functions and callbacks
    reward_functions = get_reward_functions(script_args)
    callbacks = get_callbacks(training_args, model_args, script_args)
    # Create GRPOConfig from TrainingArguments
    grpo_config = GRPOConfig(
        **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
        **{ 
        # REMOVED model_init_kwargs here 
        # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
        }
    )

    grpo_trainer = GRPOTrainer(
        model=model,                      # Our initialized Qwen model
        reward_funcs=reward_functions,    # List of reward functions from previous step
        args=grpo_config,                # GRPOConfig (created from TrainingArguments)
        train_dataset=dataset['train'],   # Training dataset
        eval_dataset=dataset['test'],    # Evaluation dataset
        callbacks=callbacks              # List of callbacks
    )
    # Start the GRPO Training Loop
    train_result = grpo_trainer.train()

    # Save the tokenizer
    tokenizer.save_pretrained(TRAINED_MODEL_PATH)

    # Save the trained model
    grpo_trainer.save_model(TRAINED_MODEL_PATH)

    print(f"GRPO Trained model saved to {TRAINED_MODEL_PATH}")
