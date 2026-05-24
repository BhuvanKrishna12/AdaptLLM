import os
import sys
import json
import time
import torch
import threading
import gradio as gr
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, PeftModel
from trl import SFTTrainer
from huggingface_hub import HfApi

from dataset_validator import validate_dataset
from dataset_formatter import format_to_alpaca
from qlora_config import get_lora_config, get_training_args

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip().replace('\r', '').replace('\n', '')

model_id = "google/gemma-3-4b-it"
HF_USERNAME = "Bhu1Krishna"

# global state
training_logs = []
training_done = False
training_error = None
adapter_repo_id = None
base_model = None
ft_model = None
tokenizer = None

def load_base():
    global base_model, tokenizer
    if base_model is not None:
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )

def run_training(file_path, repo_name):
    global training_logs, training_done, training_error, adapter_repo_id, ft_model

    training_logs = []
    training_done = False
    training_error = None

    try:
        # step 1 validate
        training_logs.append("Validating dataset...")
        ok, msg = validate_dataset(file_path)
        if not ok:
            training_error = msg
            return
        training_logs.append(f"✓ {msg}")

        # step 2 format
        training_logs.append("Formatting dataset...")
        format_to_alpaca(file_path, "../data/formatted.json")
        training_logs.append("✓ Dataset formatted to Alpaca format")

        # step 3 load model
        training_logs.append("Loading model...")
        load_base()
        training_logs.append("✓ Model loaded")

        # step 4 setup lora
        lora_config = get_lora_config()
        model = get_peft_model(base_model, lora_config)
        training_logs.append(f"✓ LoRA config applied — trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # step 5 load dataset
        with open("../data/formatted.json") as f:
            data = json.load(f)
        formatted = [{"text": f"### Instruction:\n{d['instruction']}\n\n### Input:\n{d['input']}\n\n### Response:\n{d['output']}"} for d in data]
        dataset = Dataset.from_list(formatted)
        training_logs.append(f"✓ Dataset ready — {len(dataset)} rows")

        # step 6 train
        training_logs.append("Starting training...")
        training_args = get_training_args(output_dir="../train/outputs/adapter")

        class LogCallback:
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    training_logs.append(f"  epoch {round(logs.get('epoch', 0), 2)} — loss: {round(logs['loss'], 4)}")

        from transformers import TrainerCallback
        class CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    training_logs.append(f"  epoch {round(logs.get('epoch',0),2)} — loss: {round(logs['loss'],4)}")

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer,
            callbacks=[CB()],
        )
        trainer.train()
        training_logs.append("✓ Training complete")

        # step 7 save
        model.save_pretrained("../train/outputs/adapter")
        tokenizer.save_pretrained("../train/outputs/adapter")
        training_logs.append("✓ Adapter saved")

        # step 8 push to hub
        training_logs.append("Pushing adapter to HuggingFace Hub...")
        token = os.environ.get("HF_TOKEN", "").strip().replace('\r', '').replace('\n', '')
        repo_id = f"{HF_USERNAME}/{repo_name}"
        api = HfApi()
        api.create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path="../train/outputs/adapter",
            repo_id=repo_id,
            token=token,
            repo_type="model"
        )
        adapter_repo_id = repo_id
        training_logs.append(f"✓ Adapter pushed to huggingface.co/{repo_id}")

        # step 9 load ft model
        ft_model = PeftModel.from_pretrained(base_model, "../train/outputs/adapter")
        training_logs.append("✓ Fine-tuned model ready for chat")
        training_done = True

    except Exception as e:
        training_error = str(e)
        training_logs.append(f"✗ Error: {training_error}")


def start_training(file, repo_name):
    if file is None:
        yield "Please upload a dataset first.", gr.update(visible=False), gr.update(visible=False)
        return
    if not repo_name:
        yield "Please enter a model name.", gr.update(visible=False), gr.update(visible=False)
        return

    thread = threading.Thread(target=run_training, args=(file.name, repo_name))
    thread.start()

    while thread.is_alive():
        time.sleep(1)
        logs = "\n".join(training_logs)
        if training_error:
            yield logs, gr.update(visible=False), gr.update(visible=False)
            return
        yield logs, gr.update(visible=False), gr.update(visible=False)

    logs = "\n".join(training_logs)
    hub_link = f"https://huggingface.co/{adapter_repo_id}"
    code = f"""from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")
model = PeftModel.from_pretrained(base, "{adapter_repo_id}")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")"""

    yield logs, gr.update(visible=True, value=f"Model live at: {hub_link}\n\n{code}"), gr.update(visible=True)
def generate(model, prompt, max_new_tokens=300):
    formatted = f"### Instruction:\nPlease answer the following question.\n\n### Input:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()
def chat(question):
    if base_model is None or ft_model is None:
        return "Models not loaded yet. Please complete training first.", ""
    base_resp = generate(base_model, question)
    ft_resp = generate(ft_model, question)
    return base_resp, ft_resp


with gr.Blocks(title="AdaptLLM") as demo:
    gr.Markdown("# AdaptLLM")
    gr.Markdown("Fine-tune any LLM on your own data. Upload a dataset, train, and deploy.")

    with gr.Tabs() as tabs:
        with gr.Tab("Train", id=0):
            gr.Markdown("### Step 1 — Upload your dataset")
            gr.Markdown("CSV with columns: `instruction`, `input`, `output`. Minimum 50 rows.")
            file_input = gr.File(label="Upload CSV", file_types=[".csv", ".json"])
            repo_input = gr.Textbox(label="Model name on HuggingFace", placeholder="my-medical-chatbot")
            train_btn = gr.Button("Validate & Train", variant="primary")
            log_output = gr.Textbox(label="Training logs", lines=15, interactive=False)
            result_box = gr.Textbox(label="Your model is ready", lines=8, interactive=False, visible=False)
            chat_btn = gr.Button("Open Chat Interface", variant="secondary", visible=False)

            train_btn.click(
                fn=start_training,
                inputs=[file_input, repo_input],
                outputs=[log_output, result_box, chat_btn]
            )
            def switch_to_chat():
                return gr.Tabs(selected=1)

            chat_btn.click(fn=switch_to_chat, inputs=None, outputs=tabs)

        with gr.Tab("Chat", id=1):
            gr.Markdown("### Compare base model vs your fine-tuned model")
            question = gr.Textbox(label="Your question", placeholder="Describe your symptoms...", lines=3)
            ask_btn = gr.Button("Ask", variant="primary")
            with gr.Row():
                base_out = gr.Textbox(label="Base Gemma 3 4B", lines=10)
                ft_out = gr.Textbox(label="Your fine-tuned model", lines=10)
            ask_btn.click(fn=chat, inputs=question, outputs=[base_out, ft_out])

def load_models_if_available():
    global ft_model
    adapter_path = "../train/outputs/adapter"
    if os.path.exists(adapter_path):
        print("Found existing adapter, loading...")
        load_base()
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        print("Models loaded and ready for chat")

if __name__ == "__main__":
    load_models_if_available()
    demo.launch(share=True)