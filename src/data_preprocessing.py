from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import os

# Set model and tokenizer
MODEL_NAME = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LENGTH = 256

def load_codexglue_dataset():
    dataset = load_dataset("google/code_x_glue_ct_code_to_text", "python")
    df = dataset['train'].to_pandas()
    df = df[['code', 'docstring']]  # `docstring` is the comment
    df.rename(columns={'docstring': 'comment'}, inplace=True)
    return df

def clean_text(text):
    return text.replace('\n', ' ').strip()

def tokenize_data(df):
    inputs, outputs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        code = clean_text(row['code'])
        comment = clean_text(row['comment'])

        tokenized_input = tokenizer(code, truncation=True, padding="max_length", max_length=MAX_LENGTH)
        tokenized_output = tokenizer(comment, truncation=True, padding="max_length", max_length=MAX_LENGTH)

        inputs.append({
            "input_ids": tokenized_input["input_ids"] if isinstance(tokenized_input["input_ids"], list) else tokenized_input["input_ids"].tolist(),
            "attention_mask": tokenized_input["attention_mask"] if isinstance(tokenized_input["attention_mask"], list) else tokenized_input["attention_mask"].tolist()
        })
        
        outputs.append({
            "input_ids": tokenized_output["input_ids"] if isinstance(tokenized_output["input_ids"], list) else tokenized_output["input_ids"].tolist()
        })

    return inputs, outputs


def save_processed_data(inputs, outputs, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "inputs.json"), "w") as f:
        json.dump(inputs, f)
    with open(os.path.join(out_dir, "outputs.json"), "w") as f:
        json.dump(outputs, f)

def run_pipeline():
    print("🚀 Loading dataset...")
    df = load_codexglue_dataset()
    print(f"✅ Loaded {len(df)} samples.")

    print("🔧 Cleaning + Tokenizing...")
    inputs, outputs = tokenize_data(df[:5000])  # limit to 5000 for dev

    print("💾 Saving processed data...")
    save_processed_data(inputs, outputs)
    print("🎉 Done!")

if __name__ == "__main__":
    run_pipeline()
 
