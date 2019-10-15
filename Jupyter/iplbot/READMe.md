# Rasa Command line


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
Command Line (action_endpoint should be local host in this case)
```
python -W ignore -m rasa shell --quiet --enable-api --log-file out.log --cors *
```

## Slack
-  Create a workspace, a channel, an app and a bot.
-  Re-install the app
-  Turn Event subscription ON. Subscribe to workspace events: message.channel , message.groups , message.im and message.mpim
-  Ngrok: In a separate window 
	```
	C:\Temp\ngrok.exe http 5005
	```
	Note down the token like 9bdfa563 

- Change endpoints.yml (same port as ngrok)
	```
	action_endpoint:
	 url: "http://9bdfa563.ngrok.io:5005/webhook"
	```
- Change credentials.yml file with the Slack chat bot OAuth token (starts with xoxob)

- In another window, with activate rasa environment
	```
	python -W ignore -m rasa run --connector slack --port 5005 --cors *
	```
- In Slack App Event subscription, Verify (rasa server, ngrok, actions, all must be running)
	```
	https://9bdfa563.ngrok.io/webhooks/slack/webhook
	```
	