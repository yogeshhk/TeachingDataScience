# Comments about the code
This is the code for this video series: https://www.youtube.com/watch?v=nAEb1lOf_4o

## Installing RDKIT
You will need rdkit to run this code.

Follow these instructions to install rdkit.
https://www.rdkit.org/docs/Install.html

If you run on Ubuntu / WSL you can simply run:
```
sudo apt-get install python-rdkit
```
Ideally execute the code in an anaconda environment, that's the easiest solution with rdkit.

## Installing the other packages
For pytorch geometric follow this tutorial:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Make sure your CUDA version as well as torch version match the PyG version you install.
I've used torch 1.6.0 as it seemed to be most stable with the other libraries.

## Further things
- Its highly recommended to setup a GPU (including CUDA) for this code. 
- Here is where I found ideas for node / edge features: https://www.researchgate.net/figure/Descriptions-of-node-and-edge-features_tbl1_339424976
- There is also a Kaggle competition that used this dataset (from a University):
https://www.kaggle.com/c/iml2019/overview

## Dashboard (MLFlow + Streamlit)
It is required to use conda for this setup, e.g.
```
wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
``` 

You need to start the following things:
- Streamlit server
```
streamlit run dashboard.py
```

- MlFlow Server
```
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0
    --port 5000
```

- MlFlow served model
```
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m "models:/YourModelName/Staging" -p 1234
```

TODO: Check if multi-input models work for MLFLOW!!!
