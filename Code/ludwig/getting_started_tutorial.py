# https://ludwig.ai/latest/getting_started/prepare_data/
# conda create -n ludwig python=3.10
# conda activate ludwig
# conda install cuda -c nvidia
# conda install cudatoolkit
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Check `import torch; print(torch.cuda.is_available())` Must be True
# LD_LIBRARY_PATH has paths to <cuda version>/lib x64 and Win32
# pip install bitsandbyes==0.40.2 gives error 'CUDA Setup failed despite GPU being available' need windows version
# pip install bitsandbytes-windows gives error 'no attribute 'cuDeviceGetCount''
# pip install git+https://github.com/Keith-Hon/bitsandbytes-windows gives same error
# pip install bitsandbytes==0.40.2 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
# pip install ludwig
import torch
print(torch.cuda.is_available())

from ludwig.api import LudwigModel
import pandas as pd
#
df = pd.read_csv('./data/rotten_tomatoes.csv')
config_dict = {
    "input_features": [
        {
            "name": "genres",
            "type": "set",
            "preprocessing": {
                "tokenizer": "comma"
            }
        },
        {
            "name": "content_rating",
            "type": "category"
        },
        {
            "name": "top_critic",
            "type": "binary"
        },
        {
            "name": "runtime",
            "type": "number"
        },
        {
            "name": "review_content",
            "type": "text",
            "encoder": {
                "type": "embed"
            }
        }
    ],
    "output_features": [
        {
            "name": "recommended",
            "type": "binary"
        }
    ]
}
model = LudwigModel(config=config_dict)
_ = model.train(dataset=df,output_directory="results")
model_dir = "./models/tomatoes"
model.save(model_dir)

model = LudwigModel.load(model_dir)
results = model.predict(dataset=df)
print(results)