# https://predibase.com/blog/how-to-efficiently-fine-tune-gemma-7b-with-open-source-ludwig

# Check CUDA version by `nvcc --version`, mine is 11.8
# CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
# Have corresponding CUDA based pytorch https://pytorch.org/get-started/locally/

import torch

print(torch.cuda.is_available())

from ludwig.api import LudwigModel
import pandas as pd
import yaml

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


def finetune(df, model_folder):
    model = LudwigModel(config=instruction_tuning_config_yaml)
    results = model.train(dataset=df, output_directory="results")
    model.save(model_folder)


def predict(df, model_folder):
    model = LudwigModel.load(model_folder)
    predictions_df, output_directory = model.predict(dataset=tf)
    print(predictions_df["Answer_response"].tolist())


if __name__ == "__main__":
    model_dir = "./models/gst_qna"
    train_df = pd.read_csv('./data/cbic-gst_gov_in_fgaq.csv', encoding='cp1252')

    finetune(train_df, model_dir)

    test_df = pd.DataFrame([
        {
            "Question": "What is GST?"
        },
        {
            "Question": "Does aggregate turnover include value of inward supplies received on which RCM is payable?"
        },
    ])
    # predict(test_df, model_dir)
