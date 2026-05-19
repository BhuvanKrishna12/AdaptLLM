import os
import torch
import gradio as gr
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

model_id = "google/gemma-3-4b-it"
adapter_path = "../train/outputs/adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
)

print("Loading fine-tuned model...")
ft_model = PeftModel.from_pretrained(base_model, adapter_path)

def generate(model, prompt, max_new_tokens=300):
    formatted = f"### Instruction:\nIf you are a doctor, please answer the medical questions based on the patient's description.\n\n### Input:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

def respond(question):
    base_response = generate(base_model, question)
    ft_response = generate(ft_model, question)
    return base_response, ft_response

with gr.Blocks(title="AdaptLLM — Medical Q&A") as demo:
    gr.Markdown("# AdaptLLM — Medical Q&A")
    gr.Markdown("Compare base Gemma 3 4B vs fine-tuned on ChatDoctor dataset")
    
    with gr.Row():
        question = gr.Textbox(
            label="Your medical question",
            placeholder="Describe your symptoms...",
            lines=3
        )
    
    submit_btn = gr.Button("Ask", variant="primary")
    
    with gr.Row():
        base_out = gr.Textbox(label="Base Gemma 3 4B", lines=10)
        ft_out = gr.Textbox(label="Fine-tuned on ChatDoctor", lines=10)
    
    submit_btn.click(
        fn=respond,
        inputs=question,
        outputs=[base_out, ft_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)