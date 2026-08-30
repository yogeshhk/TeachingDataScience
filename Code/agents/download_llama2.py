# https://towardsdatascience.com/two-ways-to-download-and-access-llama-2-locally-8a432ed232a4

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model_path = "./model/llama7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.float16)

# Save the model and the tokenizer to your PC
model.save_pretrained(base_model_path, from_pt=True)
tokenizer.save_pretrained(base_model_path, from_pt=True)
