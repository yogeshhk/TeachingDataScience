# https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Spacy_Custom_NER_Youtube.ipynb
# https://www.youtube.com/watch?v=YLQvVpCXpbU

import spacy
# import json
#
# nlp = spacy.load("en_core_web_sm")
# doc_sm = nlp("Donald Trump was President of USA")
# for ent in doc_sm.ents:
#     print(ent.text, ent.label_)
#
# # https://www.kaggle.com/datasets/finalepoch/medical-ner
# with open('./data/Corona2.json', 'r') as f:
#     data = json.load(f)
#
# # print(data['examples'][0]['content'])
# # print(data['examples'][0]['annotations'])
#
# training_data = []
# for example in data['examples']:
#     temp_dict = {'text': example['content'], 'entities': []}
#     for annotation in example['annotations']:
#         start = annotation['start']
#         end = annotation['end']
#         label = annotation['tag_name'].upper()
#         temp_dict['entities'].append((start, end, label))
#     training_data.append(temp_dict)
#
# # print(training_data[0])
#
# from spacy.tokens import DocBin
# from tqdm import tqdm
#
# # For each text in the training data, create a 'doc' via 'nlp-en' model. Add all such docs to docs-bin
# nlp = spacy.blank("en")  # load a new spacy model
# doc_bin = DocBin()
#
# from spacy.util import filter_spans
#
# for training_example in tqdm(training_data):
#     text = training_example['text']
#     labels = training_example['entities']
#     doc = nlp.make_doc(text)
#     ents = []
#     for start, end, label in labels:
#         span = doc.char_span(start, end, label=label, alignment_mode="contract")
#         if span is None:
#             print("Skipping entity")
#         else:
#             ents.append(span)
#     filtered_ents = filter_spans(ents)
#     doc.ents = filtered_ents
#     doc_bin.add(doc)
#
# doc_bin.to_disk("data/medical_train.spacy")

# https://spacy.io/usage/training#quickstart to create the config file
# !python -m spacy init fill-config base_config.cfg config.cfg
# !python -m spacy train config.cfg --output ./ --paths.train data/medical_train.spacy --paths.dev data/medical_train.spacy
# creates two folders 'model-last' and 'model-best'. Load best
nlp_ner = spacy.load("model-best")
doc_custom = nlp_ner("""While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in those
                 with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents like
                  loperamide are also effective at reducing the number of stools but not the duration of disease.
                [8] These agents should be used only if bloody diarrhea is not present.""")

for ent in doc_custom.ents:
    print(ent.text, ent.label_)
