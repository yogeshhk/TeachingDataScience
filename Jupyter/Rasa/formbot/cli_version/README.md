# Formbot

The `formbot` example is designed to help you understand how the `FormAction` works and how
to implement it in practice. Using the code and data files in this directory, you
can build a simple restaurant search assistant capable of recommending
restaurants based on user preferences.

## What’s inside this example?

This example contains some training data and the main files needed to build an
assistant on your local machine. The `formbot` consists of the following files:

- **data/nlu.md** contains training examples for the NLU model  
- **data/stories.md** contains training stories for the Core model
- **actions.py** contains the implementation of a custom `FormAction`
- **config.yml** contains the model configuration
- **domain.yml** contains the domain of the assistant  
- **endpoints.yml** contains the webhook configuration for the custom actions

<YK> Removed  - name: DucklingHTTPExtractor from config.yml as I did not have duckling docker 

## How to use this example?

Using this example you can build an actual assistant which demonstrates the
functionality of the `FormAction`. You can test the example using the following
steps:

- Train a Rasa model containing the Rasa NLU and Rasa Core models by running:
    ```
    python -W ignore -m rasa train
    ```
    The model will be stored in the `/models` directory as a zipped file.


- In one window:
    ```
	python -W ignore -m rasa run actions
    ```
    This will start the action server which in turn is running Forms Action

- In another window:
    ```
	python -W ignore -m rasa shell --quiet --enable-api --log-file out.log --cors * -m models --endpoints endpoints.yml
    ```
    This will load the assistant in your command line for you to chat.
	
## Sample Chat
Bot loaded. Type a message and press enter (use '/stop' to exit):
Your input ->  hi
Hello! I am restaurant search assistant! How can I help?
Your input ->  I would like a resturant
what cuisine?
Your input ->  chinese
how many people?
Your input ->  for three people?
do you want to seat outside?
Your input ->  yes
please provide additional preferences
Your input ->  no
please give your feedback on your experience so far
Your input ->  good
All done!
I am going to run a restaurant search using the following parameters:
 - cuisine: chinese
 - num_people: 3
 - outdoor_seating: True
 - preferences: no additional preferences
 - feedback: good
Your input ->  /stop
## Reference
- Article https://blog.rasa.com/building-contextual-assistants-with-rasa-formaction/?_ga=2.224850522.1350868921.1573130601-1504673344.1570291649
- Repo https://github.com/RasaHQ/rasa/tree/master/examples/formbot 
