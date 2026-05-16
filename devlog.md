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
