# 01 - Setting up the Environment

You will need
* IDE or text editor
* python 3.6 or 3.7

Setting up environment
* virtual environment
* financial demo vot
    - clone local copy
    - install dependencies
* run the bot
* telegram account
* install ngrok

## Virtual Environments

Let you scope packages to project directory instead of installing packages globally on system. Allors to use different version of same package for different projects.
They use `venv`, I'll use `conda`


## Clone financial bot repo

Use this as a backbone

This bot can 
* answer questions about account balance
* transfer funds to another account
* check spending/earning history

Clone the repo : https://github.com/RasaHQ/financial-demo.git


## Install Dependencies
we will need to install dependencies

```
pip install -r requirements.txt
```

## Install spacy language models

Then we need to install spacy language models.

```
python -m spacy download en_core_web_md
python -m spacy link en_core_web_md en
```

## Running Duckling

Duckling is a pre-trained entity extractor used to extract amounts of money, times, dates.

Runs as a separate service in its own container.

They've deployed duckling as a server already for us : http://duckling.rasa.com:8000

we'll point to this service instead of localhost.

## Start the assistant

1. Train the model (`rasa train`)
2. Start the Rasa Open Source server (`rasa shell`)
3. Start the action server (in new terminal window) (`rasa run actions`)

We will now be able to talk to the assistant through the command line.

## Creating your Telegram account

Later in the workshop, we will need to connect our assitant to the outside world using Telegram. For that w e will need a telegram acocunt
* Download Telegram on computer (or phone)
* Set up telegram account
* Login and be ready to connect our Rasa Assistant to it.



## Resources :
* Rasa Financial Demo : https://github.com/RasaHQ/financial-demo 