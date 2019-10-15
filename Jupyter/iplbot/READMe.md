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
Command Line
```
python -W ignore -m rasa shell --quiet --enable-api --log-file out.log --cors *
```

Slack
In a separate window 
```
C:\Temp\ngrok.exe http 5005
```
Note down the token like 305a4906

endpoints.yml
```
action_endpoint:
 url: "http://305a4906.ngrok.io:5055/webhook"
```

In Event subscription, Verify
```
https://305a4906.ngrok.io/webhooks/slack/webhook
```