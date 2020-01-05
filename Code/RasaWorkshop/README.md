# Rasa full-day Workshop

## ToDOs
- Instalations must be ready as per OS, no time for that during workshop
- API registrations must be done, so keep API keys ready.
- Story (google home gift, zingat, sorry, family member)
- Can do projects in Pair Programming
- Have participants read the slides to avoid monotonocity

## Queries
- Are entities and Slots same, as they seem to represent similar information?

	No. Entities are in NLU and Slots are in Core. Thats just architectural difference. But for working also they can be different.
	Say, NLU entity can be location, but can be filled into two different slots of Source location and Destination location.
	(Ref: https://forum.rasa.com/t/mapping-same-entity-to-different-slots/1389)
	
- Whats the difference between Slots Filling and Forms Actions?
	Slots are the information needed to process a request or an API. FormAction is mechanism to iterate over slots and automatically ask if any of the slots has not been filled.
	(Ref: https://blog.rasa.com/building-contextual-assistants-with-rasa-formaction/)

## Errors and Solutions
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


- Need to start slack demo in following order (imp)
	action server in window#1
	run rasa server in window #2 , on port 5004, 
	ngrok http on 5004, get the ngrok url token, 
	put it in Slack Verify then only it verifies fine else it errors. 

