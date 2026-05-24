import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

model_id = "google/gemma-3-4b-it"
adapter_path = "outputs/adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
)

def generate(prompt, max_new_tokens=200):
    formatted = f"### Instruction:\nIf you are a doctor, please answer the medical questions based on the patient's description.\n\n### Input:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

test_question = "I have been having a persistent headache for 3 days with some nausea. What could this be?"

print("\n--- BASE MODEL ---")
print(generate(test_question))

print("\nLoading adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

print("\n--- FINE-TUNED MODEL ---")
print(generate(test_question))