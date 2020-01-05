## new rasa user
* greet
  - utter_greet
* name{"name":"Alice"}
  - utter_ask_location
* location{"location":"New York"}
  - utter_used_rasa
* deny
  - utter_docs

## existing rasa user 1
* greet
  - utter_greet
* name{"name":"Tom"}
  - utter_ask_location
* location{"location":"Berlin"}
  - utter_used_rasa
* affirm
  - utter_send_blog
* subscribe
  - action_subscribe
* goodbye
  - utter_goodbye

## existing rasa user 2
* greet
  - utter_greet
* name{"name":"Ben"}
  - utter_ask_location
* location{"location":"London"}
  - utter_used_rasa
* affirm+subscribe
  - utter_send_blog
  - action_subscribe
* deny
  - utter_goodbye

## existing rasa user 3
* greet
  - utter_greet
* name{"name":"Arthur"}
  - utter_ask_location
* location{"location":"Brasil"}
  - utter_used_rasa
* affirm+subscribe
  - utter_send_blog
  - action_subscribe
* goodbye
  - utter_goodbye
