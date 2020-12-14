import re
import logging
import random
import json
import os

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
import numpy as np


START_CHAR, END_CHAR = 'start char', 'end char'
MAXLEN = 3
STEP = 1
BATCH_SIZE = 128
MAXQUESTION = 20
DATASET_PATH = 'datasets/dataset_1.txt'
EPOCH_NUMBER = 100
LEARNING_RATE = 0.01
INPUT_REPLACE_CHARS = (('„“«»', '"'), ('…', '...'), ('’', r"'"),
                       (r'(\S)([,.!?;:"\'-])', r'\1 \2'), (r'(["\'.-])(\S)', r'\1 \2'))
OUTPUT_REPLACE_CHARS = ((r'(\S) ([,.!?;:])', r'\1\2'),)
alphabet, reverse_alphabet = {}, {}
tokens = []
logs = []


def aggregate_text(temp_text, replace_chars):
    for x, y in replace_chars:
        temp_text = re.sub(x, y, temp_text)
    return temp_text


def get_alphabets(text):
    chars = sorted(set(re.split(r'\n| ', text)) | {START_CHAR, END_CHAR})
    return {key: value for key, value in enumerate(chars)}, {value: key for key, value in enumerate(chars)}


def get_tokens(text):
    return [[START_CHAR] + obj.split() + [END_CHAR] for obj in text.split('\n')]


def chars_to_int(chars):
    return list(map(lambda x: reverse_alphabet[x], chars))


def get_matrices(sentences):
    dataset = []
    for sentence in sentences:
        int_sequence = chars_to_int(sentence)
        for i in range(0, len(int_sequence) - MAXLEN, STEP):
            dataset.append(((int_sequence[i: i + MAXLEN]), (int_sequence[i + MAXLEN])))
    inputs_matrix = np.zeros((len(dataset), MAXLEN, len(alphabet)), dtype=np.bool)
    outputs_matrix = np.zeros((len(dataset), len(alphabet)), dtype=np.bool)
    for i, data in enumerate(dataset):
        for t, char in enumerate(data[0]):
            inputs_matrix[i, t, char] = 1
        outputs_matrix[i, data[1]] = 1
    return inputs_matrix, outputs_matrix


def file_record(history_data):
    index = str(len(os.listdir('training_logs')) + 1)
    filename = f'training_logs/logs-{index}.json'
    ext_data = {'maxlen': MAXLEN, 'step': STEP, 'epochs': logs, 'epochNumber': EPOCH_NUMBER, 'history': history_data,
                'datasetPath': DATASET_PATH, 'learningRate': LEARNING_RATE}
    with open(filename, 'w') as file:
        file.write(json.dumps(ext_data, indent=4, separators=(',', ': '), ensure_ascii=False,))
    model.save(f'training_models/model-{index}.h5')


logging.getLogger().setLevel(logging.INFO)
logging.info(f'-> start program with MAXLEN={MAXLEN} STEP={STEP}')
with open(DATASET_PATH, 'r') as f:
    raw_text = f.read()
new_text = aggregate_text(raw_text, INPUT_REPLACE_CHARS)
alphabet, reverse_alphabet = get_alphabets(new_text)
logging.info(f'-> created alphabet with {len(alphabet.keys())} chars')
tokens = get_tokens(new_text)
logging.info(f'-> formed tokens with {len(tokens)} length')
input_matrix, output_matrix = get_matrices(tokens)
logging.info('-> finished dataset vectorization')
model = Sequential()
model.add(LSTM(128, input_shape=(MAXLEN, len(alphabet))))
model.add(Dense(len(alphabet), activation='softmax'))
optimizer = RMSprop(learning_rate=LEARNING_RATE)
logging.info('-> finished nn')
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    logging.info('-> epoch number ' + str(epoch))
    dataset_index = random.randint(0, len(tokens) - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        logging.info('-> diversity:' + str(diversity))
        generated = []
        sentence = tokens[dataset_index][:MAXLEN]
        generated += sentence
        logging.info('-> generating with seed: "' + ' '.join(sentence) + '"')
        for i in range(MAXQUESTION):
            if END_CHAR in sentence:
                sentence = sentence[:sentence.index(END_CHAR)]
                break
            x_pred = np.zeros((1, MAXLEN, len(alphabet)))
            for t, char in enumerate(sentence[-MAXLEN:]):
                x_pred[0, t, reverse_alphabet[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = alphabet[next_index]
            sentence.append(next_char)
        str_sentence = aggregate_text(' '.join(sentence[1:]), OUTPUT_REPLACE_CHARS)
        logging.info('generated sentence: "' + str_sentence + '"')
        logs.append(str_sentence)


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
history = model.fit(input_matrix, output_matrix,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCH_NUMBER,
                    callbacks=[print_callback])
file_record(history.history)
