# Fine Tune Phi-2 Model on Your Dataset https://www.youtube.com/watch?v=eLy74j0KCrY

import gc
import os
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
login(token=hf_api_token)

# Add these imports at the top
from datetime import datetime
import logging
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Add this function to monitor system resources
def log_system_info():
    gpu_memory = torch.cuda.memory_allocated(0) / 1024 ** 2  # Convert to MB
    gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 2
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent

    logger.info(f"GPU Memory Used: {gpu_memory:.2f} MB")
    logger.info(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")
    logger.info(f"CPU Usage: {cpu_percent}%")
    logger.info(f"RAM Usage: {ram_percent}%")


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")  # Should show NVIDIA GPU name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config here
dataset_path = "D:/Yogesh/GitHub/Sarvadnya/src/fine-tuning/data/AmodMentalHealthCounselingConversations_train.csv"
formatted_path = "D:/Yogesh/GitHub/Sarvadnya/src/fine-tuning/data/AmodMentalHealthCounselingConversations_formatted.csv"
base_model = "microsoft/phi-2"
fine_tuned_model = "phi2-mental-health"

# ----------------------------------------
# dataset = load_dataset("csv", data_files=dataset_path, split="train")
# df = pd.DataFrame(dataset)
#
#
# # Each LLM has different instruction tuning prompt format
# def convert_to_llama_instruct_format(row):
#     question = row['Context']
#     answer = row['Response']
#     formatted_string = f"[INST] {question} [/INST] {answer} "
#     return formatted_string
#
#
# df['text'] = df.apply(convert_to_llama_instruct_format, axis=1)
# new_df = df[['text']] # skip other columns, 'text' is default name
# new_df.to_csv(formatted_path, index=False)
# ---------------------------------------------------

training_dataset = load_dataset("csv", data_files=formatted_path, split="train")
# Split into training and evaluation datasets
split_datasets = training_dataset.train_test_split(test_size=0.1, seed=42)  # 90% train, 10% eval

# Separate datasets
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# # Inspect datasets
# print(train_dataset.column_names)
# print(eval_dataset.column_names)

# ---
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnd_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Enable double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnd_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map={'': 0}  # Explicitly map all modules to GPU 0
)

# Ensure model parameters are on the correct device
for param in model.parameters():
    if param.device.type != 'cuda':
        param.data = param.data.to(device)

# Verify embedding layer device
if hasattr(model, 'embed_tokens'):
    model.embed_tokens = model.embed_tokens.to(device)

model.config.use_cache = False
model.config.pretraining_tp = 1

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
training_arguments = TrainingArguments(
    output_dir="./models",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=128,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    learning_rate=2e-4,  # Added explicit learning rate
    save_steps=200,
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=0.3,  # Added gradient clipping
    save_total_limit=2,  # Limit checkpoints
    fp16=True,  # Enable mixed precision training
    report_to=["tensorboard"],  # Enable TensorBoard logging
    max_steps=10,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["Wqkv", "fc1", "fc2"],
    bias="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Ensure this is provided
    peft_config=peft_config,
    # dataset_text_field="Text",
    # max_sequence=690,
    tokenizer=tokenizer,
    args=training_arguments
)


# trainer.train()


class TrainingMonitorCallback(transformers.TrainerCallback):
    def __init__(self):
        self.start_time = datetime.now()

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  # Log every 10 steps
            elapsed = datetime.now() - self.start_time
            logger.info(f"Step: {state.global_step}")
            logger.info(f"Elapsed time: {elapsed}")
            log_system_info()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Training logs: {logs}")


# Add this before starting training
trainer.add_callback(TrainingMonitorCallback())

# Start training with explicit progress monitoring
logger.info("Starting training...")
try:
    trainer.train()
except Exception as e:
    logger.error(f"Training error occurred: {str(e)}")
    log_system_info()  # Log system state when error occurs
    raise e

# Run text generation pipeline with our next model
prompt = "I am not able to sleep in night. Do you have any suggestions?"
pipe = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=250, device=device)
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

del model
del pipe
del trainer
torch.cuda.empty_cache()  # Added explicit CUDA cache clearing
gc.collect()
