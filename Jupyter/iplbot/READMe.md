# Rasa Command line Script

## Init
```
rasa init  --no-prompt
``` 
Creates project structure and dummy files. Good to run and execute
Add your data to nlu.md, stories.md and wherever needed.

## Train
```
rasa train
```

## Server
Run rasa server by
```
python -W ignore -m rasa run actions
```
"-W ignore" removes the numpy FutureWarnings

## Execution
-	Command Line (action_endpoint should be local host in this case)
	```
	python -W ignore -m rasa shell --quiet --enable-api --log-file out.log --cors *
	```

-	Local UI (Optional)
	```
	python -W ignore -m rasa x --cors *
	```
	Rasa X gives nice UI out of the box to test the bot and manage its data and conversations.


## Slack
-  Create a workspace ("DataHacksConf2019"), a channel ("#rasachatbot") and an app ("rasachatbotdemo").
-  Note down Bot user OAuth (starting with xoxb)
-  Turn Event subscription ON. Subscribe to workspace events: message.channel , message.groups , message.im and message.mpim
-  Re-install the app

- Change credentials.yml file with the Slack chat bot OAuth token (starts with xoxb) and channel ("#rasachatbot")

-  Ngrok: In a separate window 
	```
	C:\Temp\ngrok.exe http 5055
	```
	Note down the token like 3d3f77f1 

- Change endpoints.yml (same port as ngrok)
	```
	action_endpoint:
	 url: "http://3d3f77f1.ngrok.io:5055/webhook"
	```
- In another window, with activate rasa environment on a different port 5002
	```
	python -W ignore -m rasa run --connector slack --port 5002 --cors *
	```
	You will get a message like this:  Starting Rasa server on http://localhost:5002
	Now, deploy port 5002 to the internet:
	```
	C:\Temp\ngrok.exe http 5002
	```	
	Note down different ngrok token, use that below in Slack
	
- In Slack App Event subscription, Verify (rasa server, ngrok, actions, all must be running)
	```
	https://0c2fb87b.ngrok.io/webhooks/slack/webhook
	```
	
- Start chatting in Slack