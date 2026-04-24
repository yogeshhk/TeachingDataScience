# Ref: https://huggingface.co/marathi-llm/MahaMarathi-7B-v24.01-Base
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained('marathi-llm/MahaMarathi-7B-v24.01-Base')
model = LlamaForCausalLM.from_pretrained('marathi-llm/MahaMarathi-7B-v24.01-Base', torch_dtype=torch.bfloat16)

prompt = "मी एक ए. आय. द्वारा तयार केलेले महाभाषा समीकरण संच आहे."
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f"for prompt {prompt} response is {response}")
