
import tensorflow as tf
import numpy as np
import unicodedata
import re

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),
    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

# We will now split the data into two separate lists, each contains its own sentences.
# Then we will apply the functions above and add two special tokens: <start> and <end>:
# <img src="inputoutput_en_decoder.png">
# The encoder requires only sequences from source language as inputs.
# The decoder requires two versions of destination language’s sequences, one for inputs and one for targets.
raw_data_en, raw_data_fr = list(zip(*raw_data)) # unzipping
raw_data_en = [ normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en) # gives integer index to all words
data_en = en_tokenizer.texts_to_sequences(raw_data_en) # make integer index sequence of word sequence in the sentence
print(data_en)
# [[8, 5, 21, 22, 23], [24, 25, 6, 26, 27, 28, 1], [5, 29, 30, 31, 32, 9, 8, 7, 6, 1], ...]
# Need to pad zeros so that all sequences have the same length. Else, we won’t be able to create tf.data.Dataset object
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')
print(data_en)
# [[ 8  5 21 22 23  0  0  0  0  0]
#  [24 25  6 26 27 28  1  0  0  0]
#  [ 5 29 30 31 32  9  8  7  6  1],...]

#  Do the same with French sentences
fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

# We can call fit_on_texts multiple times on different corpora and it will update vocabulary automatically.
# ATTENTION: always finish with fit_on_texts before moving on
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)

data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,
                                                           padding='post')

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                            padding='post')

# We only need to create an instance of tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(20).batch(5)
print(dataset)