# Rasa Command line Script

- Init the project structure and default files
```
rasa init  --no-prompt
``` 
Creates project structure and dummy files. Good to run and execute
Replace your data in nlu.md, stories.md and wherever needed.

- Train both NLU and Dialog models together
```
rasa train
```
- Have endpoints.yml, this is for actions, a separate window, activa
	```
	action_endpoint:
		  url: "http://localhost:5055/webhook"
	```			

- Run rasa action server by
```
python -W ignore -m rasa run actions
```
"-W ignore" removes the numpy FutureWarnings

-	Command Line (action_endpoint should be local host in this case)
	```
	python -W ignore -m rasa shell --quiet --enable-api --log-file out.log --cors *
	```

-	Local UI (Optional)
	```
	python -W ignore -m rasa x --cors *
	```
	Rasa X gives nice UI out of the box to test the bot and manage its data and conversations.


- Slack
	-  Create a workspace ("DataHacksConf2019"), a channel ("#rasachatbot") and an app ("rasachatbotdemo").
	-  Note down Bot user OAuth (starting with xoxb)
	-  Turn Event subscription ON. Subscribe to workspace events: message.channel , message.groups , message.im and message.mpim
	-  Re-install the app

	- Change credentials.yml file with the Slack chat bot OAuth token (starts with xoxb) 
		```
		slack:
			slack_token: "xoxb-XXXXXXXXXXXXXXXXXXXXXXXX"
		```
	- In another window, with activate rasa environment on a different port 5004
		```
		python -W ignore -m rasa run --connector slack --port 5004 --cors *
		```
			You will get a message like this:  Starting Rasa server on http://localhost:5004
		
	- Now, deploy port 5004 to the internet:
		```
		C:\Temp\ngrok.exe http 5004
		```	
		Note down different ngrok token, aa98d86f, use that below in Slack
		
	- In Slack App Event subscription, Verify (rasa server, ngrok, actions, all must be running)
		```
		https://aa98d86f.ngrok.io/webhooks/slack/webhook
		```
		
	- Start chatting in Slack