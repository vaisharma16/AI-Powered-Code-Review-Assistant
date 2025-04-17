from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

# Raw code snippet
code_snippets = [
    """class Node:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None""",
    """def add(a, b):\n    return a + b"""
]

input_data = []
for code in code_snippets:
    prompt = "Review this Python code. Give bullet-point feedback on correctness, readability, and improvements:\n\n" + code
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    input_data.append({
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    })

with open("data/processed/inputstrial.json", "w") as f:
    json.dump(input_data, f, indent=2)

review_texts = [
    """- ✅ Correct constructor for Node class\n- 🧹 Readability is good\n- 🧠 Consider type hints for better clarity""",
    """- ✅ Simple and correct addition function\n- 💡 Could add error checking for input types"""
]

output_data = []
for text in review_texts:
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    output_data.append({
        "input_ids": tokens["input_ids"]
    })

with open("data/processed/outputstrial.json", "w") as f:
    json.dump(output_data, f, indent=2)
