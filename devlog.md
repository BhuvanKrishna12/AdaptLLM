# AdaptLLM Dev Log

## Day 1 May 16

### Goal 1:

Set up environment, load Gemma 3 4B, verify inference on RTX 4060

### What I did

- Set up WSL2 with CUDA, verified RTX 4060 visible inside WSL
- Installed transformers, peft, trl, bitsandbytes, gradio, wandb
- Fought Phi-3 cache/versioning bug, switched to Gemma 3 4B
- Accepted Gemma license on HuggingFace, authenticated via token
- Model loading and baseline inference working on GPU

### What I learned

- 4-bit quantization loads a 4B model in ~3GB VRAM
- Gemma is gated — requires license acceptance on HF before download
- WSL has a completely separate Python environment from Windows
- HF token needs to be set explicitly when CLI login doesn't persist

### What I built

- Full project folder structure
- Verified Gemma 3 4B generates coherent responses on RTX 4060

### Goal 2

Build dataset validation and formatting pipeline

### What I did

- Built dataset_validator.py, checks file type, required columns, row count, empty fields
- Built dataset_formatter.py, converts valid CSV to Alpaca JSON format
- Created sample dataset in docs/example_datasets/
- Tested both scripts end to end

### What I learned

- Alpaca format is the standard for instruction fine-tuning: instruction, input, output fields
- CSVs with commas inside text fields need proper quoting or the parser breaks
- Separating validation from formatting keeps the pipeline clean and modular

## Day 2 — May 18

### Goal

Write QLoRA config, run first fine-tuning, verify adapter saves correctly

### What I did

- Built qlora_config.py with LoRA config (r=16, alpha=32) and SFTConfig
- Switched from TrainingArguments to SFTConfig for trl 1.4.0 compatibility
- Downloaded ChatDoctor-HealthCareMagic-100k dataset (500 rows) from HF
- Debugged multiple issues: fp16 vs bf16, max_seq_length naming, OOM errors
- Fixed OOM by reducing batch size to 1, gradient accumulation to 8, max_length=512
- Full 3 epoch training run completed in ~21 minutes on RTX 4060
- Built test_inference.py for side-by-side base vs fine-tuned comparison
- Verified fine-tuned model responds in clinical doctor style vs base model's generic style

### What I learned

- Gemma uses BFloat16 not Float16 — mixing them causes a CUDA error
- SFTConfig in trl 1.4.0 uses max_length not max_seq_length
- Long medical text inputs cause OOM — capping sequence length is essential
- Fine-tuning changes response style and tone more than factual content
- LoRA only trains 0.27% of parameters (11.8M out of 4.3B) — everything else is frozen

### Results

- Train loss: 3.592 → 2.261 over 3 epochs
- Token accuracy: 0.38 → 0.52
- Adapter saved to outputs/adapter (~50MB vs 4.3GB full model)
