# Errors and Solutions
- np errors/warnings from tensorflow
	- use "python -W rasa ..." format for rasa command line. "-W" suppresses the warnings
	```
	python -W ignore -m rasa shell --quiet --enable-api --log-file out.log --cors *
	```
- Error: "NoneType" not iterable
	- Retrain all the models and use --cors flag for further commands, as shown above
	```
	rasa train
	```
- Slack Error:  error 500: " Your URL didn’t respond with the value of the challenge parameter." Bad Gateaway for ngrok window
	-??
-  Error: asyncio  - Task exception was never retrieved
	- ?? Rasa requires 2-core VM. As soon as I tested it with 2-cores VM, it resolved the issue.
- Error: If No policy ensemble or domain set. Skipping action prediction :
	- retrain the model ; rasa train



