# AdaptLLM

> Fine-tune any LLM on your own data. No ML expertise required.

A local LLM fine-tuning pipeline that trains LoRA adapters on user-provided datasets and deploys them for public inference. Built in one week as a solo project.

## Architecture

```
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
  • Code snippet generated
       ↓
Gradio Chat Interface
  • Base model vs fine-tuned side by side
  • Anyone can integrate via provided code snippet
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment

Create a `.env` file in the root folder:

```
HF_TOKEN=your_huggingface_write_token
WANDB_API_KEY=your_wandb_api_key
```

### 3. Run the unified UI

```bash
cd train
python3 ui.py
```

Upload your dataset, click Train, and get a deployed model.

## How it works

### Step 1: Prepare your dataset

CSV with three columns:

```
instruction, input, output
"Answer this question", "Your question here", "The answer here"
```

Minimum 50 rows. `input` column can be empty. See `docs/example_datasets/` for reference.

### Step 2: Train

Upload your CSV in the UI, give your model a name, click "Validate & Train". The app will:

- Validate your dataset format
- Convert to Alpaca training format
- Fine-tune Gemma 3 4B using QLoRA on your GPU
- Save the LoRA adapter
- Push it to HuggingFace Hub

### Step 3: Use your model

After training the app shows your model link and a ready-to-use code snippet:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")
model = PeftModel.from_pretrained(base, "your-username/your-model-name")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
```

## Results

### Medical Q&A: ChatDoctor dataset

| Epoch | Loss  | Token Accuracy |
| ----- | ----- | -------------- |
| 1     | 2.402 | 0.49           |
| 2     | 2.300 | 0.51           |
| 3     | 2.261 | 0.52           |

Training time: ~21 minutes on RTX 4060 8GB

**Question:** I have been taking metformin and recently started ciprofloxacin for a UTI. I am experiencing severe hypoglycemia episodes. Could these two medications be interacting?

**Base Gemma 3 4B:** In general, ciprofloxacin and metformin are not interacting. It is not clear how you are getting hypoglycemia. Please consult your physician.

**Fine-tuned on ChatDoctor:** Ciprofloxacin is a fluoroquinolone antibiotic which can affect glucose synthesis enzymes. When combined with metformin, this can lead to severe hypoglycemia. You need to stop the ciprofloxacin and switch to an antibiotic which does not affect glucose metabolism.

### Finance Q&A: Finance Alpaca dataset

| Epoch | Loss | Token Accuracy |
| ----- | ---- | -------------- |
| 1     | 2.31 | 0.51           |
| 2     | 2.18 | 0.53           |
| 3     | 2.09 | 0.54           |

Training time: ~21 minutes on RTX 4060 8GB

**Question:** What is the difference between a Roth IRA and a traditional IRA?

**Base Gemma 3 4B:** With a traditional IRA, you get a tax deduction now and pay taxes in retirement. With a Roth IRA, you don't get a deduction now but withdrawals are tax-free.

**Fine-tuned on Finance Alpaca:** A Roth IRA grows tax-free and withdrawals in retirement are tax-free. A traditional IRA grows tax-deferred and withdrawals are taxed as ordinary income. Key difference: Roth uses after-tax dollars, traditional uses pre-tax dollars.

## Published Models

| Domain      | Model                                                                               | Dataset                    |
| ----------- | ----------------------------------------------------------------------------------- | -------------------------- |
| Medical Q&A | [Bhu1Krishna/adaptllm-medical](https://huggingface.co/Bhu1Krishna/adaptllm-medical) | ChatDoctor-HealthCareMagic |
| Finance Q&A | [Bhu1Krishna/adaptllm-finance](https://huggingface.co/Bhu1Krishna/adaptllm-finance) | Finance Alpaca             |

## Project Structure

```
AdaptLLM/
├── train/
│   ├── ui.py                     # Unified training + chat interface
│   ├── train.py                  # QLoRA fine-tuning pipeline
│   ├── qlora_config.py           # LoRA and training hyperparameters
│   ├── dataset_validator.py      # Dataset format validation
│   ├── dataset_formatter.py      # CSV to Alpaca JSON converter
│   ├── test_inference.py         # Base vs fine-tuned comparison
│   └── push_to_hub.py            # Push adapter to HF Hub
├── serve/
│   ├── app.py                    # Standalone Gradio chat interface
│   └── requirements.txt
├── docs/
│   └── example_datasets/
│       ├── sample.csv            # Sample medical Q&A dataset
│       └── finance_qa.csv        # Sample finance Q&A dataset
├── devlog.md                     # Day by day build log
└── README.md
```

## Tech Stack

- **Base model:** Gemma 3 4B (google/gemma-3-4b-it)
- **Fine-tuning:** QLoRA via `peft` + `trl`
- **Quantization:** 4-bit NF4 via `bitsandbytes`
- **Experiment tracking:** Weights & Biases
- **Interface:** Gradio
- **Deployment:** HuggingFace Hub

## Requirements

- CUDA GPU with 8GB+ VRAM
- Python 3.10+
- WSL2 (if on Windows)

## Limitations

- Fine-tuning runs locally hence, a GPU is required
- 500 row dataset produces style adaptation more than deep factual improvement
- Inference also requires a GPU for reasonable speed
- Medical and finance demos are for research purposes only

## Future Work

- Cloud training backend: trigger training without local GPU
- Automatic evaluation metrics beyond loss
- Support for multiple base models
- Web-based dataset builder

## Devlog

See [devlog.md](devlog.md) for a day by day account of how this was built.
