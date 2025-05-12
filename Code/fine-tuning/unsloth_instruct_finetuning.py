from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import os


max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",  # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# We now use the Phi-3 format for conversation style finetunes. We use Open Assistant conversations in ShareGPT style.
# Phi-3 renders multi turn conversations like below:

# <|user|>
# Hi!<|end|>
# <|assistant|>
# Hello! How are you?<|end|>
# <|user|>
# I'm doing great! And you?<|end|>


tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts, }


dataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, )

print(dataset[5]["conversations"])
print(dataset[5]["text"])

unsloth_template = \
    "{{ bos_token }}" \
    "{{ 'You are a helpful assistant to the user\n' }}" \
    "{% for message in messages %}" \
    "{% if message['role'] == 'user' %}" \
    "{{ '>>> User: ' + message['content'] + '\n' }}" \
    "{% elif message['role'] == 'assistant' %}" \
    "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}" \
    "{% endif %}" \
    "{% endfor %}" \
    "{% if add_generation_prompt %}" \
    "{{ '>>> Assistant: ' }}" \
    "{% endif %}"
unsloth_eos_token = "eos_token"

tokenizer = get_chat_template(
    tokenizer,
    chat_template=(unsloth_template, unsloth_eos_token,),  # You must provide a template and EOS token
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
    map_eos_token=True,  # Maps <|im_end|> to </s> instead
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.batch_decode(outputs))

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)

model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
