# Tensorflow Text

tf.20:
    - No more sessions feed dicts
    - directly use keras as first class
    - @function used to create graph
    - session graph for production not development 
    - pre-trained from tf hub as first keras layer
	
tf.Text:
	- RaggedTensors takes care of variable length, all ops work fine.
	- Got tokenizers, 
	- All preprocessing is in graph so that in production just pass text, no need to do SIMILAR processing there as well. 
	- RNN supported.
	
## Prep Talks
- Demystifying BERT (AV article 2010/09) Use Bert client server library, Use tf hub ready Bert model on first layer. 
- Tensorflow Text (NLP with Tf)


## Notes
Noam Chomsky: Language, Cognition, and Deep Learning | Artificial Intelligence (AI) Podcast
- Is Deep Learning Engineering or Science? If it is building something useful (which it is) then its Engineering. If its helping understand something then its Science but it is not. Google Translate is Engineering. It gives you best answer based on millions of examples, but its understanding of language is 0.