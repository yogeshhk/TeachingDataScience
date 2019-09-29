# food-chatbot

This is the **Complete Version** of the Chatbot and can be directly used with slack by filling **slack_credentials.yml** and **Zomato API's** key in the **actions.py** files.

## Setup and installation

If you haven’t installed Rasa NLU and Rasa Core yet, you can do it by navigating to the project directory and running:  
```
pip install -r requirements.txt
```

You also need to install a spaCy English language model. You can install it by running:

```
python -m spacy download en
```

### Files for Rasa NLU model

- **data/nlu_data.md** file contents training data for the NLU model.
	
- **nlu_config.yml** file contains the configuration of the Rasa NLU pipeline:  
```yaml
language: "en"

pipeline: spacy_sklearn
```	

### Files for Rasa Core model

- **data/stories.md** file contains some training stories which represent the conversations between a user and the assistant. 
- **domain.yml** file describes the domain of the assistant which includes intents, entities, slots, templates and actions the assistant should be aware of.  
- **actions.py** file contains the code of a custom action which retrieves results of the latest IPL match by making an external API call.
- **endpoints.yml** file contains the webhook configuration for custom action.  
- **policies.yml** file contains the configuration of the training policies for Rasa Core model.

## How to run locally

**Note**: If running on Windows, you will either have to [install make](http://gnuwin32.sourceforge.net/packages/make.htm) or copy the following commands from the [Makefile](https://github.com/mohdsanadzakirizvi/food-chatbot/blob/master/complete_version/Makefile)

1. You can train the Rasa NLU model by running:  
```make train-nlu```  
This will train the Rasa NLU model and store it inside the `/models/current/nlu` folder of your project directory.

2. Train the Rasa Core model by running:  
```make train-core```  
This will train the Rasa Core model and store it inside the `/models/current/dialogue` folder of your project directory.

3. In a new terminal start the server for the custom action by running:  
```make action-server```  
This will start the server for emulating the custom action.

4. Test the assistant by running:  
```make cmdline```  
This will load the assistant in your terminal for you to chat.

## How to deploy to Slack

1. Go to your Slack app's settings page and use the **Bot User OAuth Access Token:** 
![](../images/bot_token.png)
And add this in the **slack_credentials.yml** file:

```python
slack:
  slack_token: "Bot User OAuth Access Token"
  slack_channel: 
```

2. Start the action server by typing the following command in terminal:

```
make action-server
```

3. Setup ngrok for the port that the action server is using by the following command:

```
ngrok http 5055
```

This will give you an output like the following:
![](../images/ngrok_action.png)

4. Copy the highlighted url in the above image into your **endpoints.yml** file:

```python
action_endpoint: "your_url_here/webhook"
  url: 
```

5. Start the core server in another terminal window:

```
python -m rasa_core.run -d models/current/dialogue -u models/current/nlu --port 5002 --connector slack --credentials slack_credentials.yml --endpoints endpoints.yml
```

This will start the server at port 5002.

6. Now you have to expose this port to the world by using ngrok, open another terminal and type:

```
ngrok http 5002
```

7. Take the above url and paste it into the **Events Subscription** page of your slack app in the following format:

```
your_url_here/webhooks/slack/webhook
```

![](../images/event_subs.png)
And you should now be able to talk to your chatbot in Slack!
