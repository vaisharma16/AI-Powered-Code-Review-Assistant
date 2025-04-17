import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/codet5-reviewer" #"Salesforce/codet5-small"      


tokenizer = AutoTokenizer.from_pretrained("models/codet5-reviewer")
model = T5ForConditionalGeneration.from_pretrained("models/codet5-reviewer")
def generate_review(code: str, max_length: int = 256) -> str:
    prompt = (
    "You are a senior software engineer. Review the following Python code and provide feedback in bullet points "
    "about correctness, readability, and improvements:\n\n"
    f"{code.strip()}"
)


    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
)
        

    review = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Clean and format the output
    review_lines = review.split("\n")
    clean_lines = [line.strip("•*- ") for line in review_lines if line.strip()]
    bullet_points = ["• " + line for line in clean_lines]
    print("🧪 Raw Model Output:", review)

    return "\n".join(bullet_points)
