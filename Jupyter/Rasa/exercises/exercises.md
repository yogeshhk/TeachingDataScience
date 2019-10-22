# Exercise

This file contains the walkthrough of the exercise to be completed using this repository.

## 1. Natural language understanding with Rasa 1.0

To train a NLU model with Rasa, you will need two things - training data and model configuration.

---

#### 1.1 Create NLU training data

You can find some training data inside the `./data/nlu.md` file. NLU training data follows the format:

```
## intent:intent_name
 - example1
 - example2
 ...
```
Add a new intent `name` with a few training examples (don't forget to annotate entities). For example, your new defined intent can look something like:  

```
## intent:name
 - My name is [Tom](name)
 - I am [Tina](name)
 - Call me [Sarah](name)
 ``` 
---

#### 1.2 Create NLU model configuration pipeline

`config.yml` file contains a sample configuration of the model pipeline. You can modify the pipeline if you
like by adding new [pipeline components](http://rasa.com/docs/rasa/nlu/choosing-a-pipeline/). 

---

#### 1.3 Train the NLU model
Once you update the training data and model configuration, train the NLU model by running:

`rasa train nlu`

Once mode is trained, it will be saved in a `./models` directory of you project. Test the model by running:

`rasa shell nlu`

---

#### 1.4 Add synonyms
Synonyms can help you normalize the extracted entity values. It's very useful if
you want to use entities to query the database or make an API call. Update the 
intent `location` by adding synonyms to some location names where applicable. For 
example:

```
## intent:location
- I am from the [United States](location:USA)
- I am from [Berlin](location)
- I am based in the [New York](location:NYC)
```

---

#### 1.5 Add out-of-scope
While it's impossible to teach your assistant to understand every possible
input, you should enable it to gracefully handle inputs it can't yet understand.
To do that, you have to first enable your assistant to identify such inputs.
This can done by defining an intent like `chitchat` which contains examples
of random user inputs.

```
## intent:chitchat
- I want pizza
- Are you a bot?
- Who made you?
```


---

#### 1.6 Add multi-intents
Another advanced use case which you should train your assistant to handle is dealing
with multi-intents - user inputs which represent more than one intention. To train 
your bot to understand such inputs, create a new multi-intent for affirmation and subscribing to the newsletter and provide some corresponding training examples. For example:

```
## intent:affirm+subscribe
- Yes. Can you subscribe me to the newsletter please?
- Yep. I would also like to become a subscriber to the Rasa newsletter.
- Yes I have. Also, add me to the Rasa newsletter subscriber list.
```

## 2. Dialogue management with Rasa 1.0

Dialogue management system is responsible of predicting how an assistant should respond based on the state of the conversation as well as context. To train a dialogue management model you will need some training stories, domain file and model configuration. You can also create custom actions which, when predicted, can call an API, retrieve data from the database or perform some other integrations.

---

#### 2.1 Create training stories
The file `./data/stories.md` already contains some training stories. Add a new example story to cover
new dialogue turn.

---

#### 2.2 Update the domain
A file called `domain.yml` contains the domain configuration of your assistant. It contains all the information an assistant needs to know to operate. Update the domain to include all newly created intents (`name`, `chitchat`, `affirm+subscribe`), entities (`name`), slots and if you like, you can add more possible response templates.

---

#### 2.3 Create model configuration
A file `config.yml` also contains the configuration of the dialogue model. You can test the performance
of [different policies](http://rasa.com/docs/rasa/core/policies/) too. 

---

#### 2.4 Implement a custom action
The responses of your assistant can go beyond simple text templates. The responses where an assistant actually takes an action (makes an API call, extracts some data from the database, etc) are called custom actions and have to implemented as a custom action class inside the file called `actions.py`. Finish implementing the custom action so that the assistant responds with a message and sends a link to a picture:

```

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests


class ActionSubscribe(Action):

     def name(self) -> Text:
         return "action_subscribe"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
         image = requests.get('http://shibe.online/api/shibes?count=1').json()[0]

         dispatcher.utter_message("Congratulatons! You have just subscribed to Rasa newsletter. As a little celebration gift, check out this cool doggo: {}".format(image))

         return []
```

---

#### 2.5 Train the dialogue model
Train the dialogue model by running:

`rasa train`

Start the server for Rasa custom actions:

`rasa run actions`

Once the server is up and runniing you can talk to your bot by running:

`rasa shell`

Rasa also comes with handy functions which allow to visualize training stories. To do that, run:

`rasa visualize`

---

#### 2.6 Add stories with multi-intents
Using the provided stories, your assistant can handle some simple conversations. Update your stories data with examples including multi-intents and test how it affects the performance of your assistant.

---

#### 2.7 Add a fallback policy
It's likely that at some point your assistant will make a mistake - the NLU or dialogue model might get uncertain about some predictions once it gets some unexpected user inputs. Once way to go around this problem is using a FallbackPolicy which invokes a fallback action if the intent recognition has a confidence below `nlu_threshold`. Update your model configuration in `config.yml` to include the fallback policy:

```
policies:
  - name: "FallbackPolicy"
    nlu_threshold: 0.3
    core_threshold: 0.3
    fallback_action_name: 'action_default_fallback'
```


## 3. Close the feedback loop with Rasa X
After you build and test a simple version of your assistant, it's important to give it to the
actual users, collect the feedback from them and improve your assistant continuously. 

---

#### 3.1 Use Rasa X to improve your assistant
The best way to improve your assistant using real conversational data is using Rasa X. In this repository,
you will find files called `rasa.db` and `tracked.db` which already contain a few real-world conversations
between our bot and the users. We will reuse them to improve the assistant using the Rasa X. Start your 
assistant with Rasa X using:

`rasa x`

In the `Conversations` tab of the Rasa X you will find existing conversations between users and your assistant.
Correct them using interactive learning and add them to the training data.

Hit the button `Train` to retrain the model and select a newly trained one inside the `Models` tab of the Rasa X UI.

Test the improved assistant by talking to it inside the `Talk to your bot` tab of the Rasa X UI or share it with 
your friends by generating and sharing a link to your bot inside the `Conversations` tab. All conversations that new
generate with your assistant will end up in `Conversations` tab for you to look into and reuse to improve your assistant.

---

#### 3.2 Expose your assistant to the outside world 
To share your assistant with friends, you will have to expose your locally running assistant to the outside world.
To do that, you can deploy your assistant on a server, or use ngrok (recommended for non-production environments only).
To do that, start ngrok by running:

`ngrok http 5002`

Copy the ngrok URL and run:

`export RASA_X_HOSTNAME=https://xxxxxx.ngrok.io; rasa x`

Once the Rasa X launches, you should be able to share your assistant with your friends around the world!


## Next steps

There is so much you can do to take this assistant to the next level (or build your own from scratch). Here are
some ideas for you of what you can do:

- Implement backend integrations using [custom actions](http://rasa.com/docs/rasa/core/actions/#custom-actions)
- Create new intents, entities and write new training stories to handle more advanced use cases.
- Connect your assistant to the most popular [messaging platforms](http://rasa.com/docs/rasa-x/get-feedback-from-test-users/#testing-channel-integrations).

