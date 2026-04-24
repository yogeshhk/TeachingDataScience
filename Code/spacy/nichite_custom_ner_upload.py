# Ref: https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Push_Spacy_Model_to_HuggingFace_Youtube.ipynb
# https://www.youtube.com/watch?v=HRGc6QFA_YU
# How To Upload Spacy Model To Hugging Face Model Hub

# To upload this big model to Hugging Face, build 'wheel' first, create one 'output' dir also
# change name in model-best/meta.json to 'custom_medical_ner'
# !python -m spacy package "model-best" "output" --build wheel
# !pip install spacy-huggingface-hub
# Create a WRITE Access Token in Hugging Face Site, Account ->Settings, copy that, and put below, when asked
# !huggingface-cli login

# from spacy_huggingface_hub import push
#
# result = push("output/en_custom_medical_ner-0.0.0/dist/en_custom_medical_ner-0.0.0-py3-none-any.whl")

# View your model here:
# https://huggingface.co/yogeshkulkarni/en_custom_medical_ner
#
# Install your model: pip install https://huggingface.co/yogeshkulkarni/en_custom_medical_ner/resolve/main
# /en_custom_medical_ner-any-py3-none-any.whl
import spacy
ner_model_hf = spacy.load('en_custom_medical_ner')
doc_hf = ner_model_hf("""While bismuth compounds (Pepto-Bismol) decreased the number of bowel movements in 
those with travelers' diarrhea, they do not decrease the length of illness.[91] Anti-motility agents 
like loperamide are also effective at reducing the number of stools but not the duration of disease.[8] 
These agents should be used only if bloody diarrhea is not present.""")

for ent in doc_hf.ents:
    print(ent.text, ent.label_)


