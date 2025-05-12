import torch
print(torch.cuda.is_available())

import pandas as pd
from ludwig.api import LudwigModel

# Define the model definition
qna_tuning_config_dict = {
    "input_features": [
        {"name": "Question", "type": "text"}
    ],
    "output_features": [
        {"name": "Answer", "type": "text"}
    ]
}

dataset_file = "./data/cbic-gst_gov_in_fgaq.csv"
# Create a Ludwig model
model = LudwigModel(qna_tuning_config_dict)

df = pd.read_csv('./data/cbic-gst_gov_in_fgaq.csv', encoding='cp1252')

# Train the model
train_stats, _, _ = model.train(df)

# Print training statistics
print(train_stats)

# Evaluate the model
eval_stats, _, _ = model.evaluate(df)

# Print evaluation statistics
print(eval_stats)

# Make predictions on new data
test_df = pd.DataFrame([
    {
        "Question": "What is GST?"
    },
    {
        "Question": "Does aggregate turnover include value of inward supplies received on which RCM is payable?"
    },
])
predictions_df, output_directory = model.predict(dataset=test_df)
print(predictions_df["Answer_response"].tolist())