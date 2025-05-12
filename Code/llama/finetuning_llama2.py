# https://colab.research.google.com/drive/12dVqXZMIVxGI0uutU6HG9RWbWPXL3vts?usp=sharing

# How to fine-tune the recent Llama-2-7b model by leveraging PEFT library from Hugging Face ecosystem, as well as
# QLoRA for more memory efficient finetuning

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
device = torch.device("cpu")

dataset_name = "timdettmers/openassistant-guanaco"
# dataset_name = 'AlexanderDoria/novel17_test' #french novels ERRORS
dataset = load_dataset(dataset_name, split="train")

# model_name = "TinyPixel/Llama-2-7B-bf16-sharded" # HUGE model
# model_name = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token