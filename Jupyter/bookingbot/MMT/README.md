# (1.) Train the nlu model
		''' python nlu_model.py '''
		
		trains NLU Model and saves the model under nlu/default/chatter folder 
		

# (2.) Train the core Model
		## Start the custom action server by running
				''' python -m rasa_core_sdk.endpoint --actions actions '''
				
		## Open a new terminal and train the Rasa Core model by running:
				''' python dialogue_management_model.py  '''
		
		saves the dialogue management model under models/dialogue
				
# (3.) Online training
			'''  python -m rasa_core.train --online -d config/domain.yml -s data/stories.md -o models/dialogue -u models/nlu/default/chatter --epochs 250 --endpoints endpoints.yml  '''
			
			Note-Make sure the custom action server is running
			
# (4.) Talking to bot
			''' python bot.py '''