# AdaptLLM

> Fine-tune any LLM on your own data. No ML expertise required.

A local LLM fine-tuning pipeline that trains LoRA adapters on user-provided datasets
and deploys them for public inference. Built in one week as a solo project.

## Demo

[Add demo video/gif here]

## Architecture

User Dataset (CSV)
↓
Local Training Pipeline (RTX 4060)
• Dataset validation
• Alpaca formatting
• QLoRA fine-tuning
• Adapter saved (~50MB)
↓
HuggingFace Hub
• Adapter weights published
↓
Gradio Chat Interface
• Base model vs fine-tuned side by side
• Anyone can chat with it

## Results — Medical Q&A (ChatDoctor dataset)

| Epoch | Loss  | Token Accuracy |
| ----- | ----- | -------------- |
| 1     | 2.402 | 0.49           |
| 2     | 2.300 | 0.51           |
| 3     | 2.261 | 0.52           |

Training time: ~21 minutes on RTX 4060 8GB

### Side by side comparison

**Question:** I have been taking metformin for type 2 diabetes and recently started
ciprofloxacin for a UTI. I am experiencing severe hypoglycemia episodes.
Could these two medications be interacting?

**Base Gemma 3 4B:**

> In general, ciprofloxacin and metformin are not interacting. Metformin is a
> biguanide which lowers blood glucose. It is not clear how you are getting
> hypoglycemia. Please consult your physician.

**Fine-tuned on ChatDoctor:**

> Ciprofloxacin is a fluoroquinolone antibiotic which is a strong inhibitor of
> glucose synthesis enzymes. When combined with metformin, this can lead to
> severe hypoglycemia episodes. You need to stop the ciprofloxacin and switch
> to an antibiotic which does not affect glucose metabolism. Consult your doctor.

## Project Structure

AdaptLLM/
├── train/
│ ├── train.py # QLoRA fine-tuning pipeline
│ ├── qlora_config.py # LoRA and training hyperparameters
│ ├── dataset_validator.py # Dataset format validation
│ ├── dataset_formatter.py # CSV to Alpaca JSON converter
│ ├── test_inference.py # Base vs fine-tuned comparison
│ └── push_to_hub.py # Push adapter to HF Hub
├── serve/
│ ├── app.py # Gradio chat interface
│ └── requirements.txt
├── docs/
│ └── example_datasets/
│ └── sample.csv # Sample medical Q&A dataset
├── devlog.md # Day by day build log
└── README.md

## Quickstart

### 1. Install dependencies

```bash
pip install transformers peft trl bitsandbytes accelerate gradio wandb datasets torch
```

### 2. Prepare your dataset

CSV with three columns:
instruction, input, output
"Answer this question", "Your question here", "The answer here"
Minimum 50 rows. See `docs/example_datasets/sample.csv` for reference.

### 3. Validate and format

```bash
python3 train/dataset_validator.py your_dataset.csv
python3 train/dataset_formatter.py your_dataset.csv
```

### 4. Fine-tune

```bash
cd train
python3 train.py
```

### 5. Run the chat interface

```bash
cd serve
python3 app.py
```

## Tech Stack

- **Base model:** Gemma 3 4B (google/gemma-3-4b-it)
- **Fine-tuning:** QLoRA via `peft` + `trl`
- **Quantization:** 4-bit NF4 via `bitsandbytes`
- **Experiment tracking:** Weights & Biases
- **Interface:** Gradio
- **Deployment:** HuggingFace Hub + Spaces

## Published Model

Fine-tuned adapter available at:
[BhuvanKrishna12/adaptllm-medical](https://huggingface.co/BhuvanKrishna12/adaptllm-medical)

## Limitations

- Fine-tuning currently runs locally but requires a CUDA GPU with 8GB+ VRAM
- 500 row dataset produces style adaptation more than factual improvement
- Medical demo is for research purposes only, not for clinical use

## Future Work

- Web UI for dataset upload and training trigger
- Cloud training backend
- Support for multiple base models
- Evaluation metrics beyond loss

## Devlog

See [devlog.md](devlog.md) for a day by day account of how this was built.
