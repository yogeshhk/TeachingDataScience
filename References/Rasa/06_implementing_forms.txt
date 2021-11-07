# 06 - Implementing Forms

Used when we need to collect specific information before running an action.

e.g.
* Authentication
* filling customer information
* etc.

Rasa Forms allow you to describe all happy paths with a single story

```
## happy path 1
U: I would like to make a money transfer
B: Towards which credit card would you like to make a payment?
U: Towards my justice bank credit card
B: How much do you want to pay
U: $100
B: The transaction has been scheduled.

## happy path 2
U: I would like to make a 100$ money transfer
B: towards which credit card would you like to make a payment?
U: Towards my justice bank credit card
B: The transaction has been scheduled
```

We can handle these two cases in 1 story only :

```md
## pay credit card happy path
* pay_cc
    - cc_payment_form
    - form{"name": "cc_payment_form"}
    - form{"name": null}

```

form actions are used to fill forms. Normally ends with `_form`. After having called the form action, you activate the form using `form{"name": "<form_name>"}`, and finally deactivate it once everything is done using `form{"name": null}`.

### FormAction

FormAction is a custom action allowing you to set the required slots and determine the behavior of the form

Main components of the form action :
* **name** : define the name fo the form action

```python

def name(self) -> Text:
    """Unique identifier of the form"""
    return "cc_payment_form"
```

* **required slots** : sets the list of required slots

```python
def required_slots(tracker: Tracker) -> List[Text]:
    """A list of required slots to fill in"""
    return ["credit_card", "amount_of_money"]
```

* **submit** : defines the output of the form action once all slots are filled in

```python
def submit(self, args) -> List[Dict]:
    """Defines what the form has to do after all slots are filled in"""
    dispatched.utter_message("The payment is confirmed")
    return [AllSlotsReset()]

```

### Forms : Custom Slot Mappings

The slot mappings define how to extract slot values form user inputs.

With Slot mappings, you can define how certain slots can be extracted from:
* entities
* intents
* support free text input
* support yes/no inputs

```python
def slot_mappings(self) -> List[Text, Union[Dict, List[Dict]]]:
    """Defines how the form extracts information to fill in slots"""
    return {"confirm": [self.from_intent(value:True, intent="affirm"), self.from_intent(value:False, intent="deny")]}

```

### Forms : Validating user input

Validate method in form action allows you to check the value of the slot against a set of values.

e.g. I would like to make a [$100] payment. (slot: `amount_of_money`)

Validate

"minimum balance" : 85
"current balance" : 500

### Custom Form Action

allows you to add new methods to a regular FormAction

```python
class CustomFormAction(FormAction):
    def name(self):
        return ""
    
    def custom_method(self, args):
        return

```

### Handling Unhappy Paths

Users are not always cooperative - They can change their mind or interrupt the form
* Solution : create stories to handle the interruptions

```md
## pay credit card unhappy path

* pay_cc
    - cc_payment_form
    - form{"name": "cc_payment_form"}
* chitchat
    - utter_chitchat
    - cc_payment_form
    - form{"name": null}
```

If the user completely changes their minds, we can cancel the form.
* Use the `action_deactivate_form` to handle that situation.

```md
## chitchat
* pay_cc
    - cc_payment_form
    - form{"name": "cc_payment_form"}
* stop
    - utter_ask_continue
* deny
    - action_deactivate_form
    - form{"name": null}
```

How does formsd affect domain and policy configuration?

* Domain :

```yml
forms:
- cc_payment_form
```

* Policy configuration :

```yml
policies:
- name: FormPolicy
```

