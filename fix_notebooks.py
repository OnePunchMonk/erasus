import json
import os

def fix_harry_potter():
    path = "notebooks/demo_remove_harry_potter_from_gpt2.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # Locate wrapper cell
    found_wrapper = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "class GPT2LastTokenWrapper" in src:
                # Replace with pass or comment
                cell["source"] = [
                    "# Removed GPT2LastTokenWrapper to allow passing labels/loss computation\n",
                    "# model = GPT2LastTokenWrapper(model)\n",
                    "print('Skipping wrapper - using raw GPT-2 model')"
                ]
                found_wrapper = True
            
            if "def tokenize_texts(" in src and "labels = input_ids[:, -1].clone()" in src:
                 # Patch to use full sequence labels
                 new_src = src.replace("labels = input_ids[:, -1].clone()  # predict last token", 
                                       "labels = input_ids.clone()  # predict full sequence\n    labels[labels == tokenizer.pad_token_id] = -100")
                 cell["source"] = new_src.splitlines(keepends=True)
                 print(f"Patched tokenize_texts in {path}")
    
    if found_wrapper:
        print(f"Fixed wrapper in {path}")
    else:
        print(f"Wrapper not found in {path}")
        
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

def fix_clip():
    path = "notebooks/clip_unlearning_demo.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "VLMUnlearner(" in src and "selector=None" not in src:
                # Add selector=None
                new_src = src.replace('strategy="gradient_ascent",', 'strategy="gradient_ascent",\n    selector=None,')
                cell["source"] = new_src.splitlines(keepends=True)
                print(f"Added selector=None to {path}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

def fix_copyright():
    path = "notebooks/copyright_removal_example.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "LLMUnlearner(" in src and "selector=None" not in src:
                # Add selector=None
                new_src = src.replace('strategy="gradient_ascent",', 'strategy="gradient_ascent",\n    selector=None,')
                cell["source"] = new_src.splitlines(keepends=True)
                print(f"Added selector=None to {path}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    fix_harry_potter()
    fix_clip()
    fix_copyright()
