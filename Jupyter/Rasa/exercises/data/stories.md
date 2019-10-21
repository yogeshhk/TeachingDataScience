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


