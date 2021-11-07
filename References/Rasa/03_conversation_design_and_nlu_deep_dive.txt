# 03 - Conversation Design and NLU Deep Dive

## Conversation Design

* Initially : Conversations as simple question / answer pairs
* Real conversations are more complex. They rely on context

## Scope Conversation
* How to get started with conversation design?
    - Define assistant's purpose
    - Leverage the knowledge of domain experts (SME)
    - Common search queries
    - FAQs and Wikis

They recommend using a prototyping tool to design conversations at first.

Also, talk to SMEs and look at real conversations to get idea of use cases and stuff.


## The Importance of Context

e.g. Context : mailing to an address

BOT (previous) : "Shall we mail letters to the same address too?

* USER : "Which one do you have on file at the moment?"
  - intent : clarification_question
  - Action : show_mailing_address
  - "BOT : "We currently send letters to ..."

VS

BOT (previous) : "Shall I also update your credit card details?"

* USER: "Which one do you have on file at the moment?"
  - intent : clarification_question
  - Action : show_mailing_address
  - "BOT : "We currently charge this credit card ..."

> Same user message, but the context helps us deal with different user actions

## Dialogue handling

* User : How much did I spend at Starbucks last week 

* USER MESSAGE + STATE(PREVIOUS ACTION) => NEXT ACTION

Recurrent model deals with selection next best action (RNN, transformer, ...)


* Rules are useful from time to time.
* Instead of providing if/else conditions for each possible conversations, you provide examples of conversations you should handle.

## Training Data

* For conversation AI, 2 types of data are needed
  - NLU needs data in the form of examples for intents
    + utterances for intent classification and entity extraction (e.g. intent:bot_challenge, intent:i_like_food)
  - Dialogue management model needs data in the form of stories
    + possible conversation scenarios that bot should handle.

Hypothetical conversations that developers define are never as good as real user data
