# Installation


## Conda installation
Install the Conda(miniconda) from https://docs.conda.io/en/latest/miniconda.html as per the OS

Check the Conda version
```
conda --version
```
conda 4.7.10

In case need to upgrade, run below command
```
conda update conda
```

## Setup in Virtual Environment

Create the virtual environment

```
conda create --name rasa python=3.7
```

Python 3.6 as different asyncio format, so better to do it in 3.7

Activate the new environment to use it
```
LINUX, macOS: conda activate botenv
WINDOWS: activate bot
```

Install latest Rasa stack.
Rasa NLU + Core is now in the single package. Do not install them separately like in past. Your mileage may vary.

```
pip install rasa
```

If you get Microsoft Build tool error, 
- Go to https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017
- Build tools for Visual Studio 19 are fine too. Select only C++ Build tools. Install.
- Re-run rasa install command 

Additionally, spaCy:
```
pip install rasa[spacy]
python -m spacy download en
python -m spacy download en_core_web_md
python -m spacy link en_core_web_md en
```
If you get linking permission error, 
- Run cmd as administrator, 
- Activate rasa env
- Do all the above spacy commands.

To show conda envs in notebook
```
conda install nb_conda_kernels
```

asyncio.run is available in Python >= 3.7, for earlier versions, there is another arrangement.

Apart from this, need to
- pip install nest_asyncio
- write following code in the first cell
```
    import nest_asyncio

    nest_asyncio.apply()
    print("Event loop ready.")
```
(Ref: https://rasa.com/docs/rasa/api/jupyter-notebooks/)

### Others
- Download ngrok from https://ngrok.com/download



