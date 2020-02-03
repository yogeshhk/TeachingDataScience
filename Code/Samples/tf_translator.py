# Reference: https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/

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

#  At every forward pass, it takes batch of sequences and initial states and returns output sequences and final states
# <img src="rnn_data_shapes-1.png">

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))

#  let’s create the decoder. Without attention mechanism, the decoder is basically the same as the encoder,
#  except that it has a Dense layer to map RNN’s outputs into vocabulary space:
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)

        return logits, state_h, state_c

# See,  in Figure 1, the final states of the encoder will act as the initial states of the decoder.
# <img src="rnn_data_shapes-2.png">

EMBEDDING_SIZE = 32
LSTM_SIZE = 64

en_vocab_size = len(en_tokenizer.word_index) + 1
encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

fr_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

source_input = tf.constant([[1, 3, 5, 7, 2, 0, 0, 0]])
initial_state = encoder.init_states(1)
encoder_output, en_state_h, en_state_c = encoder(source_input, initial_state)

target_input = tf.constant([[1, 4, 6, 9, 2, 0, 0]])
decoder_output, de_state_h, de_state_c = decoder(target_input, (en_state_h, en_state_c))

print('Source sequences', source_input.shape)
print('Encoder outputs', encoder_output.shape)
print('Encoder state_h', en_state_h.shape)
print('Encoder state_c', en_state_c.shape)

print('\nDestination vocab size', fr_vocab_size)
print('Destination sequences', target_input.shape)
print('Decoder outputs', decoder_output.shape)
print('Decoder state_h', de_state_h.shape)
print('Decoder state_c', de_state_c.shape)

'''
Source sequences (1, 8)
Encoder outputs (1, 8, 64)
Encoder state_h (1, 64)
Encoder state_c (1, 64)

Destination vocab size 107
Destination sequences (1, 7)
Decoder outputs (1, 7, 107)
Decoder state_h (1, 64)
Decoder state_c (1, 64)
'''
# Since we padded zeros into the sequences, let’s not take those zeros into account when computing the loss:
def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

optimizer = tf.keras.optimizers.Adam()
@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_states = en_states

        de_outputs = decoder(target_seq_in, de_states)
        logits = de_outputs[0]
        loss = loss_func(target_seq_out, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss

# In Inference: we will feed in the <start> token. Every next time step will take the output of the last time step as
# input until we hit the <end> token or the output sequence has exceed a specific length:
def predict():
    test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []

    while True:
        de_output, de_state_h, de_state_c = decoder(
            de_input, (de_state_h, de_state_c))
        de_input = tf.argmax(de_output, -1)
        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))

# the training loop. At every epoch, we will grab batches of data for training.
# We also print out the loss value and see how the model performs at the end of each epoch:
NUM_EPOCHS = 250
BATCH_SIZE = 5

for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)

    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out, en_initial_states)

    print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))

    try:
        predict()
    except Exception:
        continue

# Seq2Seq model with Luong attention

# encoder’s state is only passed to the first node of the decoder. For that reason, the information from the encoder
# will become less and less relevant every next time step.
# Ideally, we want all time steps within the decoder to have access to the encoder’s output.

# The alignment vector
# The alignment vector is a vector that has the same length with the source sequence
# and is computed at every time step of the decoder.
# Each of its values is the score (or the probability) of the corresponding word within the source sequence:
# So each output word has a input-seqeunce long ALignment Vector.
# It tells the decoder what to focus on at each time step.

# <img src="alignment-1.png">

# The context vector
#  It is the weighted average of the encoder’s output =  dot product of the alignment vector and the encoder’s output
# <img src="context-1.png">
# Luong-style attention uses the current decoder output to compute the alignment vector,
# whereas Bahdanau’s uses the output of the previous time step
class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)

        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)
        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment