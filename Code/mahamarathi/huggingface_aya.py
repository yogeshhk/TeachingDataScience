# Ref: https://huggingface.co/CohereForAI/aya-101
# pip install -q transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "CohereForAI/aya-101"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Marathi to English translation
tur_inputs = tokenizer.encode("Translate to English: मी एक ए. आय. द्वारा तयार केलेले महाभाषा समीकरण संच आहे.", return_tensors="pt")
tur_outputs = aya_model.generate(tur_inputs, max_new_tokens=128)
print(tokenizer.decode(tur_outputs[0]))
# Aya is a multi-lingual language model

# Q: Why are there so many languages in India?
hin_inputs = tokenizer.encode("भारत में इतनी सारी भाषाएँ क्यों हैं?", return_tensors="pt")
hin_outputs = aya_model.generate(hin_inputs, max_new_tokens=128)
print(tokenizer.decode(hin_outputs[0]))
# Expected output: भारत में कई भाषाएँ हैं और विभिन्न भाषाओं के बोली जाने वाले लोग हैं। यह विभिन्नता भाषाई विविधता और सांस्कृतिक विविधता का परिणाम है। Translates to "India has many languages and people speaking different languages. This diversity is the result of linguistic diversity and cultural diversity."

print(f"for prompt {prompt} response is {response}")
