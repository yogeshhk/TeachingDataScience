# https://predibase.com/blog/ludwig-v0-8-open-source-toolkit-to-build-and-fine-tune-custom-llms-on-your-data
# Training Llama-2-7b to do text classification with a frozen encoder and cached encoder embeddings.
# https://github.com/ludwig-ai/ludwig/tree/master/examples/llm_finetuning
import torch
print(torch.cuda.is_available())

from ludwig.api import LudwigModel
import pandas as pd
import yaml
import logging

instruction_tuning_config_yaml = yaml.safe_load("""
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

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

output_features:
 - name: Answer
   type: text

trainer:
 type: finetune
 learning_rate: 0.0003
 batch_size: 1
 gradient_accumulation_steps: 8
 epochs: 3

backend:
 type: local
""")

qna_tuning_config_yaml = yaml.safe_load("""
input_features:
 - name: Question
   type: text
   encoder:
     type: auto_transformer
     pretrained_model_name_or_path: meta-llama/Llama-2-7b-hf
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
                "pretrained_model_name_or_path": "meta-llama/Llama-2-7b-hf",
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
model = LudwigModel(config=qna_tuning_config_dict)
results = model.train(dataset=df, output_directory="results")
model_dir = "./models/gst_qna"
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