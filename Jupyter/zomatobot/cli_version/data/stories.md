## food path 1
* greet
  - utter_greet
* restaurant_search{"location": "delhi", "cuisine": "pasta"}
  - slot{"location":"delhi"}
  - slot{"cuisine":"pasta"}
  - action_restaurant_search
  - utter_did_that_help
* affirm or thanks
  - utter_gratitude
* goodbye
  - utter_goodbye

## food path 2
* greet
  - utter_greet
* restaurant_search{"location": "lucknow", "cuisine": "burger"}
  - slot{"location":"lucknow"}
  - slot{"cuisine":"burger"}
  - action_restaurant_search
  - utter_did_that_help
* deny
  - utter_ask_again
* restaurant_search{"location": "chennai", "cuisine": "mughlai"}
  - slot{"location":"chennai"}
  - slot{"cuisine":"mughlai"}
  - action_restaurant_search
  - utter_did_that_help
* affirm or thanks
  - utter_gratitude

## food path 3
* restaurant_search{"location": "chennai", "cuisine": "mughlai"}
  - slot{"location":"chennai"}
  - slot{"cuisine":"mughlai"}
  - action_restaurant_search
  - utter_did_that_help
* affirm or thanks
  - utter_gratitude

## greet path
* greet
  - utter_greet

## goodbye path
* goodbye
  - utter_goodbye

## food path restart
* restaurant_search{"location": "chennai", "cuisine": "tandori"}
    - slot{"cuisine": "tandori"}
    - slot{"location": "chennai"}
    - action_restaurant_search
    - utter_did_that_help
* deny
    - utter_ask_again
* restaurant_search{"cuisine": "pasta", "location": "chennai"}
    - slot{"cuisine": "pasta"}
    - slot{"location": "chennai"}
    - action_restaurant_search
    - utter_did_that_help
* thanks
    - utter_gratitude
* goodbye
    - utter_goodbye