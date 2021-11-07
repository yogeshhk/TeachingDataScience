# 08 - Recap

### What we've learned so far
* setup
* deep dive of NLU and dialogue management
* build a MVP assistant
* Share assitant with outside world
* Make iterative improvements and take your assitant to the next level

### Pipeline
1. user says something
2. utterance goes through connector modules (fb messenger, live chat, ...)
3. NLU extracts intents and entities from utterance
4. Dialogue management decides what to do next (connect to external database, api, say something to user, ...)
5. Response selector selects best answer (sometimes, not always)
6. we output the message through the same communication channel to the user.

### Intents and Entities

* two of the most common and necessary types of information to extract from a message
    - Intent : what the user wants to achieve
    - Entity : Key piece of information
    - ex. How much did I spend at Starbucks last week? (intent : amount_spent, entity : starbucks (vendor_name))

### Policies
* Decide which action to take at every step of the conversation
* Each policy predicts an *action* with some *probability*. This is called *core confidence*
    - policy with highest confidence is the one that decides which action to take.

### Domain
* Defines the "world" of your assistant. What it knows, can understand and can do.
    - `intents` and `entities` : what your bot can understand
    - `slots` : what your bot can remember
    - `actions` : what your bot can do (and say)
    - `responses` : what your bot can say (and display)
    - `session_config` : what your bot considers a new conversation

### Multiple Policies
* The policy with *highest confidence* wins
* If the confidence of two policies is equal, the policy with highest priority wins
* SOme are rule-based, some are ml-based


### DIET
* new sota nn architecture for NLU
* predicts intents and entities together
* plug and play pretrained language models (bert, glove, ...) really customizable

### TED
* policy based on transformer architecture
* able to untangle sub-dialogues by paying attention to turns or ignoring some dialogue turns in order to decide what to do next

### Scope Conversation
* Get started with conversation design
    - assistant's purpose
    - leverage knowledge of domain experts
    - common search queries
    - FAQs and wikis

### Training Data

* NLU needs data in the form of examples for intents
* Dialogue management model needs data in the form of stories

### Minimum Viable Assistant
* A basic assistant that can handle the most important *happy path* stories
    - **Happy path** : if your assistant asks a user for some information and the user provides it, we call that a happy path
    - **Unhappy path** : All the possible edge cases of the bot

### Testing
* End-to-end evaluation : Run through test conversations to make sure that both NLU and Core make correct predictions
    - `rasa test`
* nlu evaluation : Split data into a test set or estimate how well your model generalizes using cross-validation
    - `rasa test nlu -u data/nlu.md --config config.yml --cross-validation`
* core evaluation : Evaluate your trained model on a set of test stories and generate a confusion matrix
    - `rasa test core --stories test_stories.md --out results`


### Custom Actions

* Triggered by the dialogue management model
* Connect to outside world (e.g. db, api, ...)

### Events

* Internally, all conversations are represented as a sequence of events
* some events are automatically tracked.
* e.g. `UserUttered`, `BotUttered`, `SlotSet`, `Restarted`, ...


### Tracker

* Tracker maintains the *state of a dialogue* between the assistant and the user. 
* keeps track of events, slots, session info, ...

### Slots

* Your bot's memory
* key-value pairs
* Can store
    - user provided info
    - info comming from outside world
* Can be set by
    - NLU (from extracted entities, buttons)
    - Custom Actions
* Can be configured to affect or not affect the dialogue progression

### Actions

* Things your bot runs in response to user input 

* 4 types:
    - Utterance actions (`utter_`)
    - Retrieval actions (`respond_`)
    - Custom actions (`action_`)
    - Default actions (`action_restart`, `action_default_fallback`, ...)

### Custom Actions (examples)

* send multiple messages
* query a db
* make an api call to another service
* return events (set a slot, force a follow up action, revert a user utterance)

### Forms

* gathering user informations needed to complete a task.

```md
* greet
    - utter_greet
* pay_cc
    - cc_payment_form
    - form{"name": "cc_payment_form"}
    - form{"name": null}
```

Not all forms are *happy paths*, can deal with in within stories

we can use `action_deactivate_form` to handle situations where users decide not to proceed with the form.

### Rasa X

* turns conversations into training data
* used to review, annotate, improve your bot

### Real conversations > Synthetic data

> The best training data is generated from the assistant's *actual conversations*

### Conversation-driven development
1. Share
2. Review
3. Annotate
4. Test
5. Track
6. Fix

### How Rasa fits into your stack

* computation layer (python, tensorflow, spacy, numpy, scikit-learn)
* covnersational AI infrastructure layer (Rasa OS)
* Tool layer (Rasa X)
* Application layer (contextual assistants)
