# 05 - Events, Slots, and Custom Actions

Events, Slots and Custom actions are related to Dialogue Management. They're not the model that decides what to do, but information that is stored of that executes stuff.

## Project setup : Files

* `__init__.py` : an empty file that helps python find your actions.
* `actions.py` : code for your custom actions
* `config.yml` : configuration of your NLU and Core models
* `credentials.yml` : details for connecting to other services
* `data/nlu.md` : your NLU training data
* `data/stories.md` : your stories
* `domain.yml` : your assistant's domain
* `endpoints.yml` : details for connecting to channels like fb messenger
* `models/<timestamp>.tar.gz` : your initial model

## Rasa Events and Slots

### Events

* *internally*, conversations are represented as a sequence of *events*
* some events are automatically tracked

Events could be : `UserUttered`, `SlotSet`, `BotUttered`, `Restarted`, ... 

### Tracker

* Trackers maintain the *state of the dialogue* between the assistant and the user
* It keeps track of events, slots, session info, etc.

### Slots

* Your bot's memory
* They are key-stored values.

* They can store
    - user-provided information
    - information from outside world (through custom actions for example)
* Can be set by
    - NLU (from extracted entities or buttons)
    - Custom Actions
* Can be configured to affect or not the dialogue progression

### Slot Types

* Text and List slots influence conversation path based on **whether they are set or not**

    - `Text` : "I am based in **London**" -> slots: location: type: text . Extracts 1 value : London

    - `List` : "Are there available appointments on **Monday** or **Wednesday** this week?" -> slots: appointment_time: type: list . Extracts 2 values : Monday and Wednesday

* Categorical, Boolean, and Float slots influence conversation path based on **the value of the slot**
    - `Categorical` : "I am looking for a restaurant in a **low** price range." -> slots: price_level: type: categorical , values: (low, medium, high)
    - `Boolean` : e.g. if user if authenticated -> slots: is_authenticated: type: bool (true, false)
    - `Float` : e.g. "Find me a restaurant within 2 mile radius" -> slots: radius: type: float, min_value: 0.0, max_value: 50.0

* Unfeaturized slots don't have any influence on the dialogue

    -`Unfeaturized` : "My name is **Sarah**" -> slots: patient_name: type: unfeaturized

Recommendation : always start with `Unfeaturized`, and change when we're sure a slot should be a certain type.

### Slots

There are a few different ways how slots can be set

* Slots set from NLU 
    - e.g. how much did I spend at [Target](vendor_name)
    - what is my typical spending at [Starbucks](vendor_name)

* Slots set by clicking buttons
* Slots set by custom actions
    - action_account_balance
    - slot{"account_balance": 1000}
    - slot{"amount_transferred": None}

Now let's define the slots for out Financial Bot assistant

## Custom Actions

Connecting to the outside world!

### Actions

Functions your bot runs in response to user input
* 4 different action types:
    - **Utterance actions** : `utter_`
        + send a specific message to the user
        + specified in `responses` section of the domain
    - **Retrieval actions** : `respond_`
        + send a message selected by a retrieval model
    - **Custom actions** : `action_`
        + run arbitrary code and send any number of messages (or none)
        + return events
    - **Default actions** :
        + built-in implementations available but can be overriden
        + e.g. `action_listen`, `action_restart`, `action_default_fallback`

### Custom Actions

Custom action code is run by a **webserver** called by the **action server**

* **how custom actions get run**
    - when a custom action is predicted, Rasa will call the **endpoint** to your action server
    - The action server
        + runs the code for your custom action
        + (optionally) returns information to modify the dialogue state
* **how to create an action server**
    - you can create an action server in any language you want ! (not only python)
    - you can use the **Rasa-SDK** for easy deployment in Python

### Custom Actions : Examples

They can do whatever you want!

* Send messages to user
* Query a database
* make an API call to another service
* return events (e.g.)
    - set a slot (e.g. based on database query)
    - force a follow up action (e.g. force a specific action to be executed next)
    - revert a user utterance (i.e. remove a user utterance from the tracker)

### Custom Actions : Query a database
Using the Rasa SDK

```python
class ActionCheckAddress(Action):
    def name(self) -> Text:
        return "action_check_address"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        idnum = tracker.get_slot("person_id")
        q = "SELECT Address FROM Customers WHERE CustomerID=1;'{0}'".format(idnum)
        result = db.query(q)

        return [SlotSet("address", result if result is not None else "NotOnFile)]

```