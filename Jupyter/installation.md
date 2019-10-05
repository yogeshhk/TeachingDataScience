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
conda create --name rasa python=3.6
```

Activate the new environment to use it
```
LINUX, macOS: conda activate botenv
WINDOWS: activate bot
```

Install latest Rasa stack
Rasa NLU
```
pip install rasa_nlu
```

For dependencies
spaCy+sklearn (pipeline)
```
pip install rasa_nlu[spacy]
python3 -m spacy download en
python3 -m spacy download en_core_web_md
python3 -m spacy link en_core_web_md en
```
If you get Microsoft Build tool error, do following and re-run the above commands

Go to https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017

Build tools for Visual Studio 19 are fine too. Select only C++ Build tools (4.53 GB)



Tensorflow (pipeline)
```
pip install rasa_nlu[tensorflow]
```

### Create Project Structure
For training/data files, we create a data directory under BASE_DIR and create the training file nlu.md in that.
Also have stories.md there. Sample files are in data dir.

In BASE_DIR have nlu_config.yml like below:
```
language: "en"
pipeline: "spacy_sklearn"
```

