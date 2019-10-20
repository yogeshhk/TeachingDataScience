# Zomato Food Ordering ChatBot 
(Reference: https://github.com/kunalj101/food-chatbot/tree/master/complete_version)


## Speficiation files

- **data/nlu.md** file contents training data for the NLU model.
- **data/stories.md** file contains some training stories which represent the conversations between a user and the assistant. 
- **config.yml** file contains the configuration of the Rasa NLU pipeline and policies
- **domain.yml** file describes the domain of the assistant which includes intents, entities, slots, templates and actions the assistant should be aware of.  
- **actions.py** file contains the code of a custom action which retrieves results by making an external API call.
- **endpoints.yml** file contains the webhook configuration for custom action.  

## How to run locally

Window#1:
- activate rasa env
- Train both NLU and Dialog models together by 
```
rasa train
```
- Models zip having both NLU and Core gets stored inside "models" directory

Window#2:
- Add your ZOMATO_API_KEY to Environment variable, then open a new window and activate rasa env
- Have endpoints.yml, this is for actions and nothing else, so its port (default 5055) can be different.
	```
	action_endpoint:
		  url: "http://localhost:5055/webhook"
	```			

- Run rasa action server in a separate window, activate rasa env, then 
```
python -W ignore -m rasa run actions
```
"-W ignore" removes the numpy FutureWarnings

Window#1
- Go back to window#1
- Start command line chatbot by 
```
python -W ignore -m rasa shell --quiet --enable-api --log-file out.log --cors *	
```

## Sample chat:
	-> Hi
	<- Hey! What can I do for you?
	-> I am hungry
	<-
			*Here are top results for north indian in gurgoan*
		Restaurant: Krispy Krunchy Chicken
		Cuisines: Fast Food
		Address: 1934 Hwy. 53 North, Gurdon 71743
		Rating: 0
		Average cost for two: 10

		Restaurant: Sonic Drive-In
		Cuisines: Fast Food
		Address: 508 North Elm Street
		Rating: 0
		Average cost for two: 20

		Restaurant: Sonic Drive-In
		Cuisines: Fast Food
		Address: 508 North Elm Street, Gurdon
		Rating: 0
		Average cost for two: 0

		*I hope that solved your query!*
	-> Yes
	<-
		Glad that I could be of help to you!
		Bye
	->  /stop
	