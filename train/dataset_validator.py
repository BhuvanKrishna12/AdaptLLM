import pandas as pd
import os

REQUIRED_COLUMNS = {"instruction", "input", "output"}
MIN_ROWS = 50
SUPPORTED_EXTENSIONS = [".csv", ".json"]

def validate_dataset(file_path: str) -> tuple[bool, str]:
    
    # check file exists
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    # check extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Use .csv or .json"
    
    # load file
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path)
    except Exception as e:
        return False, f"Could not read file: {str(e)}"
    
    # check columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing required columns: {missing}. Need: instruction, input, output"
    
    # check row count
    if len(df) < MIN_ROWS:
        return False, f"Too few rows: {len(df)}. Minimum is {MIN_ROWS}"
    
    # check empty fields
    for col in REQUIRED_COLUMNS:
        empty = df[col].isnull().sum() + (df[col] == "").sum()
        if empty > 0:
            return False, f"Column '{col}' has {empty} empty values"
    
    return True, f"Dataset valid. {len(df)} rows ready for training."


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/example_datasets/sample.csv"
    ok, msg = validate_dataset(path)
    print("✓" if ok else "✗", msg)