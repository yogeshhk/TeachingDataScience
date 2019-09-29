
## fallback
- utter_unclear


## Story 1
* flight
    - utter_boarding
* inform{"location": "BOM"}
    - action_save_origin
    - slot{"from": "BOM"}
    - utter_destination
* inform{"location": "DEL"}
    - action_save_destination
    - slot{"to": "DEL"}
    - utter_date
* inform{"date": "20-01-2019"}
    - action_save_date
    - slot{"date": "20-01-2019"}
    - utter_confirm
* affirmation
    - action_get_flight
	- utter_check_another_one
* deny
	- utter_thanks
	- action_restart

## Stry 2-multiple steps
* flight
    - utter_boarding
* inform{"location": "PNQ"}
    - action_save_origin
    - slot{"from": "PNQ"}
    - utter_destination
* inform{"location": "BLR"}
    - action_save_destination
    - slot{"to": "BLR"}
    - utter_date
* inform{"date": "03-02-2019"}
    - slot{"date": "03-02-2019"}
    - action_save_date
    - slot{"date": "03-02-2019"}
    - utter_confirm
* affirmation
    - action_get_flight
    - utter_check_another_one
* affirmation
	- action_slot_reset
	- reset_slots
    - utter_boarding
* inform{"location": "BOM"}
    - action_save_origin
    - slot{"from": "BOM"}
    - utter_destination
* inform{"location": "DEL"}
    - action_save_destination
    - slot{"to": "DEL"}
    - utter_date
* inform{"date": "10-02-2019"}
    - slot{"date": "10-02-2019"}
    - action_save_date
    - slot{"date": "10-02-2019"}
    - utter_confirm
* affirmation
    - action_get_flight
    - utter_check_another_one
* deny
    - utter_thanks
    - action_restart
