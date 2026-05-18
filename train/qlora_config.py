from peft import LoraConfig, TaskType
from trl import SFTConfig

def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

def get_training_args(output_dir="outputs/adapter"):
    return SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
    run_name="adaptllm-gemma-run1",
    max_grad_norm=0.3,
    dataloader_pin_memory=False,
    max_length=512,
    dataset_text_field="text",
)