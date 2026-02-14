import json
import glob
import os

files = glob.glob("notebooks/executed_*.ipynb")
print(f"Found {len(files)} executed notebooks: {files}")

for fpath in files:
    print(f"\n--- Analyzing {fpath} ---")
    try:
        with open(fpath, encoding="utf-8") as f:
            nb = json.load(f)
        
        has_error = False
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "error":
                        has_error = True
                        print(f"ERROR in Cell {i}:")
                        print("\n".join(out.get("traceback", [])))
        if not has_error:
            print("✅ No errors found!")
            
    except Exception as e:
        print(f"Failed to parse {fpath}: {e}")
