'''
An example of text generation using the script for
Star Wars: The Empire Strikes Back (Episode V)

The following tutorial was referenced and is thus somewhat similar in implementation:
https://www.tensorflow.org/tutorials/text/text_generation
'''

__author__ = 'Sam Prokopchuk'

import os.path
import numpy as np
import tensorflow as tf
import time

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

# Data prep constants:

# sequence length, including "X" & "Y"
# eg: if 4, the word "hello" would become the following:
'''
["h",        X       Y
 "e",      ["h",   ["e",
 "l",   =>  "e", :  "l",
 "l",       "l",    "l",
 "o" ]      "l" ]   "o"]
'''
SEQUENCE_LEN = 100


# Model params:
EMBEDDING_DIM = 256
RNN_BATCH_SIZE = 1
RNN_UNITS = 1024


# Training constants:
# Num of dataset values to shuffle in buffer
BUFFER_SIZE = 10000
EPOCHS = 10
BATCH_SIZE = 32


def get_dataset_and_info(file_path):
    def get_text():
        abspath = os.path.abspath(file_path)
        with open(abspath, 'rb') as f:
            return f.read().decode(errors='ignore')

    text = get_text()
    vocab = sorted(set(text))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = np.array(vocab)
    text_as_ids = np.array([char2id[c] for c in text])

    def seq_to_supervised(seq):
        X = seq[:-1]
        Y = seq[1:]
        return X, Y

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_ids)
    char_sequences = char_dataset.batch(SEQUENCE_LEN + 1, drop_remainder=True)
    dataset = char_sequences.map(seq_to_supervised)

    info = {
        'vocab': vocab,
        'char2id': char2id,
        'id2char': id2char,
        'text_as_ids': text_as_ids
    }

    return dataset, info


def build_model(vocab_size):
    model = Sequential()
    model.add(Embedding(
        vocab_size, EMBEDDING_DIM,
        batch_input_shape=[RNN_BATCH_SIZE, None]))
    model.add(LSTM(
        RNN_UNITS,
        return_sequences=True,
        stateful=True
    ))
    model.add(Dense(vocab_size))
    return model


def main():
    ds, info = get_dataset_and_info('./data/sw_esb_4th.txt')
    ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    vocab, char2id, id2char, text_as_ids = (
        info['vocab'],
        info['char2id'],
        info['id2char'],
        info['text_as_ids'])

    model = build_model(len(vocab))


if __name__ == '__main__':
    main()
