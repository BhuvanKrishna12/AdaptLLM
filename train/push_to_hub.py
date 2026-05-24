import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

def push_adapter(adapter_path: str, repo_id: str):
    api = HfApi()
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Adapter pushed to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    push_adapter(
        adapter_path="outputs/adapter",
        repo_id="BhuvanKrishna12/adaptllm-medical"
    )