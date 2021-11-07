# 04 - Deep dive into DIET, TED, and some testing!

## Testing

* Still need to keep these good software engineering practices even if we deal with an AI application
* unit testing
* NLU tests
* end-to-end testing
* CI/CD pipeline to automate the testing.

### Validation tests

* validate the date
  - `rasa data validate stories --max-history 5`
  - It checks for stories that are in conflict
  - the max history flag is not necessary.

* Evaluating the NLU model
  - train/test split
    + `rasa data split nlu`
  - and evaluate on test set
    + `rasa test nlu -u train_test_split/test_data.md`
    + (this outputs you a confusion matrix and creates reports from the evaluation also)

* end-to-end testing
  - `rasa test`
  - this outputs a confusion matrix, but also F1 scores, precision, accuracy, recall, ...
  - also evaluates stories (end2end)


## Deep Dive into DIET

Last years have been great for NLP
* GLUE is out, SuperGLUE is in
* Sesame Street models are in (BERT, ...)
* not much changes for every task using these large scale language models


BUT!
* large-scale language models are not ideal for conversation applications 
  - long time to train
  - iteration is slow...

### DIET is a neural network architecture for NLU

What is DIET?
* new sota nn architecture for NLU
* predicts intents and entities together
* plug and play pretrained language models (as featurizers)

DIET is Modular!
* raw text is fed to featurizers
    - tokenizers to split text into tokens
    - featurizers (sparse and dense) transform the tokenized text into vectors to be fed into DIET
        + featurizers could be character n-gram (sparse), word-level featurizers (sparse), large scale language models (dense), word2vec (dense)
    - sparse features go through a feedforward layer
    - dense features and transformed sparse features are concatenated, and then fed into a feedforward layer again before being fed into DIET.

* How to use DIET in rasa project
    - add in config file
        + define tokenizer (e.g. ConveRTTokenizer)
        + define featurizers (e.g. ConveRTFeaturizer, Glove, BERT, sparse features)
        + define classifier (DIETClassifier)
          * can tune the hyperparameters here based on our application

* paper they published about DIET on different intent entity tasks
  - NLU-benchmark dataset
  - without pretrained embeddings (sparse features only), DIET outperformed previous SOTA
  - Glove features outperformed BERT features using DIET
  - ConveRT features outperformed everyone (only trained on conversational data instead of everything)
  - DIET + ConveRT + sparse features outperformed finetuned BERT on dataset. Also 6 times faster to train!

* Which featurizers is best for me?
  - depends on the task! it's not a "one size fits all" approach.
    + they provide sensible defaults
    + these models are easy to customize so feel free to play around here!


### Transformer Embedding Dialogue Policy (TEDPolicy)

Conversation AI requires NLU and Dialogue Management

Rules don't scale beyond a certain point!

* Happy paths are already solved (easy with rule-based systems also!)
* Real conversations don't follow happy path! We want to be able to handle that!
  - we should test your model with real user conversations ASAP!

Normally, dialogue management is done using RNNs with some persistent memory (e.g. LSTM). The hidden state is what is being fed into the next step.

BUT! Not all input should be treated equally! We want to focus on the important messages (e.g. *paying attention* to the important messages ;) )

* They found that the transformer embedding dialogue policy can untangle subdialogues


### Conversation Driven Development (CDD)

* The challenge
  - When developing assistants, it's impossible to anticipate all of the things your user might say
* The approach
  - A user-centric process : listening to your users and using those insights to improve your AI assistant.

see blog post : https://blog.rasa.com/conversation-driven-development-a-better-approach-to-building-ai-assistants/ 

Steps :
1. Share (with test users)
2. Review (what the users have said and where the bots failed)
3. Annotate (annotate user inputs and add them to your bot training data)
4. Test (check if you have broken something along the way)
5. Track (changes through GitHub)
6. Fix (based on what broke)

This can all be done through Rasa X !

Successful production-ready bots follow engineering best practices 
* version control
* testing
* ci/cd

From zero to production ready deployment in under 5 minutes
* create a vm
* curl -s http://get-rasa-x.rasa.com | sudo bash
this will run in about 4 mins. This will install k3s, helm and a helm chart with everything

## Resources
* TED Policy Blog Post : https://blog.rasa.com/unpacking-the-ted-policy-in-rasa-open-source/
* Rasa Algorithm Whiteboard Youtube Playlist : https://www.youtube.com/watch?v=wWNMST6t1TA&list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb&ab_channel=Rasa
* DIET Classifier Blog Post : https://blog.rasa.com/introducing-dual-intent-and-entity-transformer-diet-state-of-the-art-performance-on-a-lightweight-architecture/
