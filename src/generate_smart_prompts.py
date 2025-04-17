import os
import json
import ast

def extract_functions_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        source = file.read()
    
    functions = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name
                docstring = ast.get_docstring(node) or ""
                body = ast.get_source_segment(source, node)
                functions.append((name, docstring, body))
    except SyntaxError:
        print(f"❌ Skipping {filepath} (invalid Python syntax)")
    
    return functions

def build_prompt(name, docstring, code):
    return f"""You are a senior software engineer. Review the following function:

Function Name: {name}
Docstring: {docstring or '[No docstring]'}
Code:
```python
{code}
```"""

# Folder where .py files are located
code_dir = "examples/code_snippets"

# Output lists
raw_inputs = []
raw_outputs = []

# Iterate over all Python files in the folder
for filename in os.listdir(code_dir):
    if filename.endswith(".py"):
        filepath = os.path.join(code_dir, filename)
        functions = extract_functions_from_file(filepath)
        for name, doc, code in functions:
            prompt = build_prompt(name, doc, code)
            raw_inputs.append(prompt)
            raw_outputs.append("")

# Save the generated smart prompts
os.makedirs("data/raw", exist_ok=True)
with open("data/raw/raw_inputs.json", "w", encoding="utf-8") as f:
    json.dump(raw_inputs, f, indent=2)
with open("data/raw/raw_outputs.json", "w", encoding="utf-8") as f:
    json.dump(raw_outputs, f, indent=2)

print(f"✅ Extracted {len(raw_inputs)} prompts from functions!")
