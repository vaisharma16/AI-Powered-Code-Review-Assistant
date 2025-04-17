import os
import json

# Folder containing your .py files
CODE_DIR = "examples/code_snippets"

# Output paths
RAW_INPUTS_PATH = "data/raw/raw_inputs.json"
RAW_OUTPUTS_PATH = "data/raw/raw_outputs.json"

# Store all code samples and placeholder reviews
raw_inputs = []
raw_outputs = []

for root, _, files in os.walk(CODE_DIR):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read().strip()
                if code:
                    raw_inputs.append(code)
                    raw_outputs.append("Review: [Add your review here]")

# Save the collected data
os.makedirs("data/raw", exist_ok=True)
with open(RAW_INPUTS_PATH, "w", encoding="utf-8") as f:
    json.dump(raw_inputs, f, indent=2)
with open(RAW_OUTPUTS_PATH, "w", encoding="utf-8") as f:
    json.dump(raw_outputs, f, indent=2)

print(f"✅ Saved {len(raw_inputs)} code examples to raw_inputs.json and raw_outputs.json")
