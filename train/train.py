import os
import torch
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model
from trl import SFTTrainer
import wandb
import json

from qlora_config import get_lora_config, get_training_args

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

model_id = "google/gemma-3-4b-it"

def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )

    return model, tokenizer


def load_dataset(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    # format into prompt strings
    formatted = []
    for item in data:
        prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        formatted.append({"text": prompt})

    return Dataset.from_list(formatted)


def train(data_path: str = "../data/formatted.json"):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="adaptllm", name="gemma-run1")

    model, tokenizer = load_model_and_tokenizer()

    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(data_path)
    print(f"Dataset loaded: {len(dataset)} rows")

    training_args = get_training_args()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving adapter...")
    model.save_pretrained("outputs/adapter")
    tokenizer.save_pretrained("outputs/adapter")
    print("Done. Adapter saved to outputs/adapter")

    wandb.finish()


if __name__ == "__main__":
    train()