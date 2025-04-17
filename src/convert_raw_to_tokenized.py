import json
from transformers import AutoTokenizer
import os

# Load raw data
with open("data/raw/raw_inputs.json", "r") as f:
    raw_inputs = json.load(f)

with open("data/raw/raw_outputs.json", "r") as f:
    raw_outputs = json.load(f)

# Initialize tokenizer
MODEL_NAME = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Prepare tokenized input-output pairs
inputs, outputs = [], []

for code, review in zip(raw_inputs, raw_outputs):
    prompt = f"Review this Python code. Give bullet-point feedback on correctness, readability, and improvements:\n\n{code}"

    tokenized_input = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    tokenized_output = tokenizer(review, padding="max_length", truncation=True, max_length=128)

    inputs.append({
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"]
    })

    outputs.append({
        "input_ids": tokenized_output["input_ids"]
    })

# Save tokenized files
os.makedirs("data/processed", exist_ok=True)

with open("data/processed/inputs.json", "w") as f:
    json.dump(inputs, f, indent=2)

with open("data/processed/outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("✅ Tokenization complete. Saved to data/processed/")
