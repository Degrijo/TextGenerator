from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import numpy as np
import random
import sys
import json


with open('./questions.txt', 'r') as f:
    text = f.read().lower()

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def get_text():
    model = load_model('model.h5')
    print('----- Generating text')
    start_index = random.randint(0, len(text) - maxlen - 1)
    diversity = 0.2
    print('----- diversity:', diversity)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    for i in range(1000):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        generated += next_char
    # output = generated.split('?')[1:-1]
    # for sentence in output:
    #     for i in range(len(sentence)):
    #         if sentence[i].isalpha():
    #             sentence = sentence[i].upper() + sentence[i:] + '?'
    #             break
    # print(output)
    return generated
