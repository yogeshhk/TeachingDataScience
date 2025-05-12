# https://predibase.com/blog/fine-tuning-mistral-7b-on-a-single-gpu-with-ludwig
import torch

print(torch.cuda.is_available())

from ludwig.api import LudwigModel
import pandas as pd
import yaml
import logging

instruction_tuning_config_yaml = yaml.safe_load("""
model_type: llm
# base_model: meta-llama/Llama-2-7b-hf
# base_model: mistralai/Mistral-7B-v0.1
base_model: alexsherstinsky/Mistral-7B-v0.1-sharded

quantization:
 bits: 4

adapter:
 type: lora

prompt:
  template: |
    ### Instruction:
    You are a taxation expert on Goods and Services Tax used in India.
    Take the Input given below which is a Question. Give Answer for it as a Response.

    ### Input:
    {Question}

    ### Response:

input_features:
 - name: Question
   type: text
   preprocessing:
      max_sequence_length: 1024
      
output_features:
 - name: Answer
   type: text
   preprocessing:
      max_sequence_length: 1024

trainer:
  type: finetune
  epochs: 5
  batch_size: 1
  eval_batch_size: 2
  gradient_accumulation_steps: 16  # effective batch size = batch size * gradient_accumulation_steps
  learning_rate: 2.0e-4
  enable_gradient_checkpointing: true
  learning_rate_scheduler:
    decay: cosine
    warmup_fraction: 0.03
    reduce_on_plateau: 0

generation:
  temperature: 0.1
  max_new_tokens: 512
  
backend:
 type: local
""")

qna_tuning_config_yaml = yaml.safe_load("""
input_features:
 - name: Question
   type: text
   encoder:
     type: auto_transformer
     pretrained_model_name_or_path: alexsherstinsky/Mistral-7B-v0.1-sharded
     trainable: false
   preprocessing:
     cache_encoder_embeddings: true

output_features:
 - name: Answer
   type: text
""")

qna_tuning_config_dict = {
    "input_features": [
        {
            "name": "Question",
            "type": "text",
            "encoder": {
                "type": "auto_transformer",
                "pretrained_model_name_or_path": "alexsherstinsky/Mistral-7B-v0.1-sharded",
                "trainable": False
            },
            "preprocessing": {
                "cache_encoder_embeddings": True
            }
        }
    ],
    "output_features": [
        {
            "name": "Answer",
            "type": "text"
        }
    ]
}

df = pd.read_csv('./data/cbic-gst_gov_in_fgaq.csv', encoding='cp1252')
model = LudwigModel(config=qna_tuning_config_dict, logging_level=logging.INFO)
results = model.train(dataset=df, output_directory="results")
model_dir = "./models/gst_mistral"
model.save(model_dir)

test_df = pd.DataFrame([
    {
        "Question": "What is GST?"
    },
    {
        "Question": "Does aggregate turnover include value of inward supplies received on which RCM is payable?"
    },
])
model = LudwigModel.load(model_dir)
predictions_df, output_directory = model.predict(dataset=test_df)
print(predictions_df["Answer_response"].tolist())
