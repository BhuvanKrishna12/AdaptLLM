import pandas as pd
import json
import os

def format_to_alpaca(file_path: str, output_path: str = "data/formatted.json") -> tuple[bool, str]:
    
    # load file
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_json(file_path)

    # convert to alpaca format
    alpaca_data = []
    for _, row in df.iterrows():
        alpaca_data.append({
            "instruction": str(row["instruction"]).strip(),
            "input": str(row["input"]).strip(),
            "output": str(row["output"]).strip()
        })

    # save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(alpaca_data, f, indent=2)

    return True, f"Formatted {len(alpaca_data)} rows → {output_path}"


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "docs/example_datasets/sample.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/formatted.json"
    ok, msg = format_to_alpaca(input_path, output_path)
    print("✓" if ok else "✗", msg)