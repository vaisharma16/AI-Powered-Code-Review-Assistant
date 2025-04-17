import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os

MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 256

# Load raw files (already uploaded by you)
with open("data/raw/raw_inputs.json") as f:
    inputs = json.load(f)

with open("data/raw/raw_outputs.json") as f:
    outputs = json.load(f)

tokenized_inputs = []
tokenized_outputs = []

for code, comment in tqdm(zip(inputs, outputs), total=len(inputs)):
    prompt = f"review: {code.strip()}"

    input_enc = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_LEN)
    output_enc = tokenizer(comment.strip(), padding="max_length", truncation=True, max_length=MAX_LEN)

    tokenized_inputs.append({
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"]
    })

    tokenized_outputs.append({
        "input_ids": output_enc["input_ids"]
    })

# Save as proper JSON (not strings)
os.makedirs("data/processed_t5small", exist_ok=True)

with open("data/processed_t5small/inputs.json", "w") as f:
    json.dump(tokenized_inputs, f, indent=2)

with open("data/processed_t5small/outputs.json", "w") as f:
    json.dump(tokenized_outputs, f, indent=2)

print("✅ Fixed JSON format for training.")
