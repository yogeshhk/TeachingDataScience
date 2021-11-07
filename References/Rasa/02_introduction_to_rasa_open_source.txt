# 02 - Introduction to Rasa Open Source

## Day 1
* Welcome
* Intro to Rasa
* Conversation Design
* Deep dive into NLU
* Deep dive into dialogue polciies
* CDD and testing
* Demo

## Day 2

* Reviewing previous day
* Events and Slots
* Custom actions and Forms
* Improving assistant with Rasa X
* Connecting assistant to external messaging channels
* What's next

## Instructors 

* Mady Mantha (Sr. Technical Evangelist)
* Juste Petraityte (Head of Developer Relations)
* Karen White (Developer Marketing Manager)

## Intro to Rasa

5 Levels of Conversational Assitants
* Command-based (last 10 years) - Users figure out how the bot works.
* Chatbot FAQs (most common today) - dialogue management not handled by ML
* Contextual Assistants (a few examples today) - Assistant understands context (what has been said before), start using CDD here. Learn from previous conversations
* Consultative Assistants (in 2-5 years) - Assistant gets to know user over time.
* Adaptative Assistants (in 5-10 years) - Fully automated CDD process. Being proactive in care of patient (e.g. you fly from berlin to montreal, assistant asks if you need a uber when you land)

## Rasa Open Source

* Rasa NLU for intent classification and entity extraction
* Rasa Core for ML-based dialogue management
* NLG replaced by Response Selection in their slides (using response + utterance to make choice)


## Real conversations don't follow happy paths.

Goal of the workshop is to build Minimum Viable Assistant with Rasa Open Source + improve it using Rasa X

(You've seen that slide! ;) )

## NLU

Subset of NLP (Natural Language Processing). Responsible of understanding the meaning of text (unstructured data)

ex. How much did I spend at Starbucks last week?

* Intent : search_transaction (what is the user trying to achieve)

* Entity : vendor_name, time (key piece of information)

### How it works under the hood (input and output)

1. Raw text as user input (how much did I spend at Starbucks last week?)
2. Tokenizer -> tokens ([how, much, did, i, spend, at, starbucks, last, week])
3. Text Featurizer -> Text representation (dense or sparse) ([(0, 421)  1, (1, 48)  2, ..., (2, 90)  1, (2, 150)  1])
4. NLU
  - Entity Extractor -> Entities (vendor_name : starbucks)
  - Intent Classifier -> Intent (search_transactions)

### Text Representation

* Featurizers create representations of data intestible by ml models (makes us able to train sequence models)


### Text Representation : Word Vectors
* "Judge a word by the company it keeps" (word embeddings)


### Text representation : How models use word vectors

* non-sequence model :
    - one feature vector per input (whole message)
    - order of words not captured (BOW)
* Sequence models:
    - one feature vector per token (word) & feature vector for whole message
    - word order captured

NLU now can be handled in one step using DIET from word embeddings obtained (featurizers)

## Configuration files

Let's start coding!

* create sandbox branch

### Config.yml file:
* she explains that this is where you set up your NLU pipeline and policies for dialogue management

* tokenizers (whitespace, ...)
* Featurizers (Regex, lexicalsyntactic, countVectors (word level and character level), ...)
* Can also add custom components in here (sentiment analysis, spell checker, ...)

* Classifier (DIET for dual intent entity transformer) takes as input any sparse or dense featurizers and classifies it for intents and entities together

* Entity extraction can be rule-based or ML-based (e.g. duckling for dates and stuff

* Different rule-based policies (fallback, form, mapping, ...)
* ML-based policy TEDPolicy can learn from our conversational data

## Creating NLU data

* 5-10 utterances per intent is good enough to start with, and then add to it with test users
* you don't want to over-engineer the NLU data with synthetic data. Will use real data further along the way.

## Policies
* Decide which action to take at every step in the conversation
* each policy predicts an **action** with some **probability**. This is called **core confidence**

## Multiple Policies
* Policy with highest confidence wins
* if the confidence of two policies is equal, the policy with highest priority wins

priorities :
5. FormPolicy
4. FallbackPolicym TwoStageFallbackPolicy
3. MemoizationPolicy, AugmentedMemoizationPolicy
2. MappingPolicy
1. EmbeddingPolicy, KerasPolicym TEDPolicy

Rule-based policiues have higher priority than ML-based policies

## Resources
* Rasa Masterclass Youtube Playlist : https://www.youtube.com/watch?v=rlAQWbhwqLA&list=PL75e0qA87dlHQny7z43NduZHPo6qd-cRc
* L3-AI Conference Youtube Playlist : https://www.youtube.com/watch?v=bAkToyQhWyo&list=PL75e0qA87dlGP51yZ0dyNup-vwu0Rlv86&ab_channel=Rasa
* Rasa Docs : https://rasa.com/docs/
* Building Assistants Tutorial : https://rasa.com/docs/rasa/chitchat-faqs#building-assistants
* NLP for Developers Youtube Playlist : https://www.youtube.com/watch?v=fqwrGzsYAi8&list=PL75e0qA87dlFJiNMeKltWImhQxfFwaxvv
