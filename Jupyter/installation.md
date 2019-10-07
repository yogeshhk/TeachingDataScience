# Installation

(Ref: Build a Conversational Chatbot with Rasa Stack and Python— Rasa NLU - Romil Jain)

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


### Create Project Structure
- For training/data files, we create a ``data'' directory under BASE_DIR and create the training file nlu.md in that.
Also have stories.md there. 
- Create ``config'' dir in BASE_DIR, have nlu_config.yml like below:
```
language: "en"
pipeline: "spacy_sklearn"
```
- Create ``models'' directory inside BASE_DIR


### Others
- Download ngrok from https://ngrok.com/download