# https://levelup.gitconnected.com/no-more-hard-coding-use-declarative-configuration-to-build-and-fine-tune-custom-llms-on-your-data-6418b243fad7
import torch
print(torch.cuda.is_available())

from datasets import load_dataset
import pandas as pd
import yaml
import logging
from ludwig.api import LudwigModel



config_str ="""
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

quantization:
  bits: 4

adapter:
  type: lora

prompt:
  template: >-
    Below is an instruction that describes a task, paired with an input
    that may provide further context. Write a response that appropriately
    completes the request.

    ### Instruction: {instruction}

    ### Input: {input}

    ### Response:

generation:
  temperature: 0.1 # Temperature is used to control the randomness of predictions.
  max_new_tokens: 512

preprocessing:
  global_max_sequence_length: 512
  split:
    type: random
    probabilities:
    - 1
    - 0
    - 0

input_features:
  - name: instruction
    type: text

output_features:
  - name: output
    type: text

trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    warmup_fraction: 0.01
"""

config = yaml.safe_load(config_str)
dataset= load_dataset("ronal999/finance-alpaca-demo", split="train")
dataset_df = dataset.to_pandas()

model = LudwigModel(config=config, logging_level=logging.INFO)
results = model.train(dataset=dataset_df)
print(results)