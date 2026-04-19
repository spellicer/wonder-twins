import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wonder import TRAINED_MODEL_PATH, device, device_map, test_trained_model_inference

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
    torch_dtype=torch.bfloat16, # Keep the same dtype as training for consistency
    device_map=device_map
)

# Test the model
test_input = "how are you?"
response = test_trained_model_inference(trained_model, tokenizer, device, test_input)
print(f"Test Input: {test_input}")
print(f"Trained Model Response: {response}")
