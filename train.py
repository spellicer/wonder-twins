# Import necessary libraries
import logging
import os
import sys
import re
import math
from dataclasses import dataclass, field
from typing import List, Optional

# Import PyTorch and Hugging Face Transformers
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import get_last_checkpoint

# Import dataset utilities
import datasets
from datasets import load_dataset

# Import libraries from TRL (Transformers Reinforcement Learning)
from trl.experimental.ppo import (
    AutoModelForCausalLMWithValueHead, 
    PPOConfig, 
    PPOTrainer, 
)
from trl import (
    GRPOTrainer, 
    GRPOConfig, 
    SFTTrainer
)

# Import math-related utilities
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
# Load the "AI-MO/NuminaMath-TIR" dataset from DigitalLearningGmbH
MATH_le = load_dataset("AI-MO/NuminaMath-TIR", "default")  

# Access the first sample in the training set
MATH_le['train'][0]
# Load the "Bespoke-Stratos-17k" dataset from bespokelabs
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default") 

# Access the first sample in the training set
bespoke_rl['train'][0]
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-GRPO-training" # For saving our trained model

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# Initialize base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16
)

print(f"Model parameters: {model.num_parameters():,}")
# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the appropriate device
model.to(device)

# Test basic inference
def test_model_inference(user_input: str):
    """Test basic model inference with the loaded model and tokenizer."""
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and generate
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model
test_input = "how are you?"
response = test_model_inference(test_input)
print(f"Test Input: {test_input}")
print(f"Model Response: {response}")
# DeepSeek system prompt for GRPO based training
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
# Function to structure the training data
def make_conversation(example):
    """Convert dataset examples into conversation format."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }
# Load and prepare dataset
def load_math_dataset():
    """Load and prepare the mathematics dataset."""
    dataset = load_dataset(
        "AI-MO/NuminaMath-TIR",
        name="default",
        split=['train', 'test']
    )
    
    # Convert splits into dictionary
    dataset = {
        'train': dataset[0],
        'test': dataset[1]
    }
    
    # Apply conversation format
    for split in dataset:
        dataset[split] = dataset[split].map(make_conversation)

        # Remove 'messages' column if exists
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    return dataset
# Load our training dataset and printing train/test size
dataset = load_math_dataset()

print(f"Train set size: {len(dataset['train'])}")
print(f"Test set size: {len(dataset['test'])}")
def validate_dataset(dataset):
    """Perform basic validation checks on the dataset."""
    
    # Define the required fields for the dataset
    required_fields = ["problem", "prompt"]

    # Loop through the 'train' and 'test' splits of the dataset
    for split in ['train', 'test']:
        print(f"\nValidating {split} split:")

        # Retrieve column names from the dataset
        fields = dataset[split].column_names

        # Check if any required fields are missing
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"Warning: Missing fields: {missing}")  # Warn if fields are missing
        else:
            print("✓ All required fields present")  # Confirm all fields are present

        # Retrieve the first sample from the dataset split
        sample = dataset[split][0]

        # Extract the 'prompt' field, which contains a list of messages
        messages = sample['prompt']

        # Validate the prompt format:
        # - It should contain at least two messages
        # - The first message should be from the 'system' role
        # - The second message should be from the 'user' role
        if (len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'):
            print("✓ Prompt format is correct")  # Confirm correct format
        else:
            print("Warning: Incorrect prompt format")  # Warn if format is incorrect

# Validate dataset
validate_dataset(dataset)
def accuracy_reward(completions, **kwargs):
    """
    Reward function to check if the model's response is mathematically 
    equivalent to the ground truth solution.
    Uses latex2sympy2 for parsing and math_verify for validation.
    """
    
    # Extract responses
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    solutions = kwargs.get("solution") # Get solutions from kwargs
    
    for content, sol in zip(contents, solutions):
        # Parse the ground truth solution
        gold_parsed = parse(sol, extraction_mode="first_match", 
                            extraction_config=[LatexExtractionConfig()])
        
        if gold_parsed:  # Check if parsing was successful
            # Parse the model's answer with relaxed normalization
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # Reward 1.0 if correct, 0.0 if incorrect
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If ground truth cannot be parsed, assign neutral reward (0.5)
            reward = 0.5
            print("Warning: Failed to parse gold solution:", sol)

        rewards.append(reward)
    
    return rewards
# Implement Format Reward Function
def format_reward(completions, **kwargs):
    """
    Reward function to check if the completion has the correct format:
    <think>...</think> <answer>...</answer>.
    """
    # Define the regex pattern for the desired format
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    # Extract the content from each completion
    completion_contents = [completion[0]["content"] for completion in completions]

    # Check if each completion matches the pattern
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE)
               for content in completion_contents]

    # Reward 1.0 for correct format, 0.0 otherwise
    return [1.0 if match else 0.0 for match in matches]
def reasoning_steps_reward(completions, **kwargs):
    r"""
    Reward function to encourage clear step-by-step reasoning.
    It looks for patterns like "Step 1:", numbered lists, bullet points,
    and transition words.
    """
    # Regex pattern to find indicators of reasoning steps
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # Extract completion contents
    completion_contents = [completion[0]["content"] for completion in completions]

    # Count the number of reasoning step indicators in each completion
    matches = [len(re.findall(pattern, content, re.MULTILINE))
               for content in completion_contents]

    # Reward is proportional to the number of reasoning steps, maxing out at 1.0
    # We're using a "magic number" 3 here - encourage at least 3 steps for full reward
    return [min(1.0, count / 3) for count in matches]
# Implement Cosine Scaled Reward Function
def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function. This function scales the accuracy reward
    based on completion length. Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """
    def cosine_scaled_reward(completions, solution, accuracy_rewards, **kwargs):
        """
        Cosine scaled reward function that adjusts accuracy rewards based on completion length.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol, acc_reward in zip(contents, solution, accuracy_rewards):
            gen_len = len(content)  # Length of the generated answer
            progress = gen_len / max_len # How far we are to max length
            cosine = math.cos(progress * math.pi) # Cosine value based on progress

            if acc_reward > 0.5: # Assuming accuracy_reward gives ~1.0 for correct answers
                min_value = min_value_correct
                max_value = max_value_correct
            else: # Incorrect answer
                min_value = max_value_wrong  # Note the swap!
                max_value = min_value_wrong

            # Cosine scaling formula!
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards
    return cosine_scaled_reward
def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    Returns a repetition penalty reward function. Penalizes repetitions of n-grams
    in the generated text.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """Helper function to generate n-grams from text."""
        words = text.lower().split() # Lowercase and split into words
        return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        Repetition penalty reward function.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "": # No penalty for empty completions
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size: # No penalty for short completions
                rewards.append(0.0)
                continue

            ngrams = set() # Use a set to store unique n-grams
            total = 0
            for ng in zipngram(completion, ngram_size): # Generate n-grams
                ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
                total += 1 # Count total n-grams

            # Calculate scaling factor: more repetition -> higher scaling
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty # Apply penalty based on scaling
            rewards.append(reward)
        return rewards
    return repetition_penalty_reward
# Define GRPOScriptArguments for reward function parameters
@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Minimum reward for cosine scaling for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward for cosine scaling for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.8,
        metadata={"help": "Minimum reward for cosine scaling for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for cosine scaling for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for cosine scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-0.1,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
# Define TrainingArguments from transformers
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,         # Output directory for checkpoints and logs
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
)
@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """
    model_name_or_path: str = field(
        default=MODEL_NAME, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: Optional[str] = field(
        default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "Override the default `torch_dtype` and load the model under this dtype."}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading model and tokenizer."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation to use. 'flash_attention_2' or None"}
    )
# Instantiate configuration objects
script_args = GRPOScriptArguments()
model_args = ModelConfig()
# Utility function to get reward functions based on script arguments
def get_reward_functions(script_args):
    """
    Returns a list of reward functions based on the script arguments.
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward,  # Assuming accuracy_reward is defined in previous steps
        "format": format_reward,      # Assuming format_reward is defined in previous steps
        "reasoning_steps": reasoning_steps_reward, # Assuming reasoning_steps_reward is defined
        "cosine": get_cosine_scaled_reward( # Assuming get_cosine_scaled_reward is defined
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward( # Assuming get_repetition_penalty_reward is defined
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list
logger = logging.getLogger(__name__)

class LoggingCallback(TrainerCallback):
    """
    A simple callback for logging training information at specific steps.
    """
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            if state.log_history and len(state.log_history) > 0:
                logger.info(f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, Learning Rate = {state.log_history[-1].get('learning_rate', None)}")
            else:
                logger.info(f"Step {state.global_step}: No logging information available yet")

def get_callbacks(training_args, model_args, script_args):
    """
    Returns a list of callbacks to be used during training.
    For now, it includes only the LoggingCallback. You can extend this to add more callbacks.
    """
    callbacks = [LoggingCallback()] # Instantiate our LoggingCallback
    return callbacks
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
# Define the path to your trained model (same as OUTPUT_DIR)
TRAINED_MODEL_PATH = "data/Qwen-GRPO-training"

# Save the tokenizer
tokenizer.save_pretrained(TRAINED_MODEL_PATH)

# Save the trained model
grpo_trainer.save_model(TRAINED_MODEL_PATH)

print(f"GRPO Trained model saved to {TRAINED_MODEL_PATH}")
# Load the tokenizer - make sure to use trust_remote_code=True if needed
tokenizer = AutoTokenizer.from_pretrained(
    TRAINED_MODEL_PATH,
    trust_remote_code=True, # If your model config requires it
    padding_side="right" # Ensure consistent padding side
)

# Set pad token if it wasn't saved or loaded correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the trained model itself
trained_model = AutoModelForCausalLM.from_pretrained(
    TRAINED_MODEL_PATH,
    trust_remote_code=True, # If your model architecture requires it
    torch_dtype=torch.bfloat16 # Keep the same dtype as training for consistency
)

# Move the loaded model to your device (GPU if available)
trained_model.to(device) # 'device' is still our CUDA device from before
# Testing Inference with the Trained Model
def test_trained_model_inference(user_input: str):
    """Test inference with the loaded trained model and tokenizer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # Re-use our system prompt
        {"role": "user", "content": user_input}
    ]

    # Apply chat template using our tokenizer
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate output using our *trained_model*
    outputs = trained_model.generate(
        **inputs,
        max_new_tokens=200, # Maybe generate a bit longer now
        do_sample=True,
        temperature=0.7
    )

    # Decode the generated tokens back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model
test_input = "how are you?"
response = test_trained_model_inference(test_input)
print(f"Test Input: {test_input}")
print(f"Trained Model Response: {response}")
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
    overwrite_output_dir=True,
    num_train_epochs=1,         # Adjust epochs as needed
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,        # Adjust learning rate for SFT
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="no",
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
model_args = ModelConfig(
    model_name_or_path=MODEL_NAME,
    model_revision="main",
    torch_dtype="bfloat16",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
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
    tokenizer=tokenizer,                 # Tokenizer
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