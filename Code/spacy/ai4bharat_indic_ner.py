# Train #datacentric #spacy #nlp #NER model on Hindi Annotated Data
# https://www.youtube.com/watch?v=fT-9I81F990&list=PLM2ypVVB68X-7XxBamx088XEC1uTwJu7c
# https://huggingface.co/datasets/ai4bharat/naamapadam/tree/main/data https://ai4bharat.iitm.ac.in/naamapadam/
# https://github.com/astutic/acharya-spacy/tree/Spacy-Marathi-NER
# https://github.com/astutic/acharya-spacy/tree/Spacy-hindi-ner

import sys
import json


def convert_json_to_iob(infile, outfile, max_records=10):
    records = []
    with open(infile, 'r', encoding="utf8") as fi:
        for line in fi:
            record = json.loads(line)
            records.append(record)

        if max_records != -1:  # -1 means take ALL
            records = records[:max_records]

    with open(outfile, 'w', encoding="utf8") as fo:
        for i, record in enumerate(records):
            # fo.write(f"-DOCSTART-\t{i}\n\n")
            words = record['words']
            ner_tags = record['ner']
            for i, word in enumerate(words):
                tag = ner_tags[i]
                write_line = "".join([word, "\t", tag, "\n"])
                fo.write(write_line)
            fo.write("\n")


if __name__ == "__main__":
    language_train_ifilename = "./data/hi_IndicNER/hi_train.json"
    language_train_ofilename = "./data/hi_IndicNER/hi_train.iob"
    convert_json_to_iob(language_train_ifilename, language_train_ofilename,800)

    language_test_ifilename = "./data/hi_IndicNER/hi_test.json"
    language_test_ofilename = "./data/hi_IndicNER/hi_test.iob"
    convert_json_to_iob(language_test_ifilename, language_test_ofilename, 80)

    language_val_ifilename = "./data/hi_IndicNER/hi_val.json"
    language_val_ofilename = "./data/hi_IndicNER/hi_val.iob"
    convert_json_to_iob(language_val_ifilename, language_val_ofilename, 80)

# https://spacy.io/usage/training#quickstart to create the config file, Select Hindi in dropdown, download as hindi_base_config.cfg
# !python -m spacy init fill-config hindi_base_config.cfg hindi_config.cfg
# !python -m spacy convert data/hi_IndicNER/hi_train.iob data/hi_IndicNER/ -c conll
# !python -m spacy convert data/hi_IndicNER/hi_val.iob data/hi_IndicNER/ -c conll
# !python -m spacy train hindi_config.cfg --output ./ --paths.train data/hi_IndicNER/hi_train.spacy --paths.dev data/hi_IndicNER/hi_val.spacy
