 
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer

MODEL_NAME = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CodeCommentDataset(Dataset):
    def __init__(self, input_file, output_file):
        with open(input_file, 'r') as f:
            self.inputs = json.load(f)
        with open(output_file, 'r') as f:
            self.outputs = json.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]['input_ids']
        attention_mask = self.inputs[idx]['attention_mask']
        labels = self.outputs[idx]['input_ids']

        # Replace pad tokens with -100 for loss masking
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

def train():
    print("📚 Loading dataset...")
    dataset = CodeCommentDataset("data/processed/inputs.json", "data/processed/outputs.json")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"✅ Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="./models/codet5-reviewer",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print("🚀 Training started...")
    trainer.train()
    print("🎯 Training complete. Saving model...")

    model.save_pretrained("models/codet5-reviewer")
    tokenizer.save_pretrained("models/codet5-reviewer")
    print("✅ Model saved to models/codet5-reviewer")

if __name__ == "__main__":
    train()
