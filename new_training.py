from re import sub
import logging


START_CHAR, END_CHAR = 'start char', 'end char'
MAXLEN = 2
STEP = 1


alphabet, reverse_alphabet = {}, {}
dataset = []
sentences = []


def aggregate_text(raw_text):
    temp_text = sub(r'„“«»', '"', raw_text.lower())
    temp_text = sub(r'…', '...', temp_text)
    temp_text = sub(r'’', '\'', temp_text)
    temp_text = sub(r'(\S)([,.!?;:"\'\-])', r'\1 \2', temp_text)
    temp_text = sub(r'(["\'.\-])(\S)', r'\1 \2', temp_text)
    return temp_text


def get_alphabets(text):
    text = sub(r'\n', ' ', text)
    chars = sorted(list(set(text.split(' '))) + [START_CHAR, END_CHAR])
    return {key: value for key, value in enumerate(chars)}, {value: key for key, value in enumerate(chars)}


def get_dataset(text):
    collection = text.split('\n')
    for i in range(len(collection)):
        sentence = [START_CHAR] + collection[i].split(' ') + [END_CHAR]
        collection[i] = list(map(lambda smb: reverse_alphabet[smb], sentence))
    return collection


logging.getLogger().setLevel(logging.INFO)
logging.info(f'-> start program with MAXLEN={MAXLEN} STEP={STEP}')
with open('./new_questions.txt', 'r') as f:
    raw_text = f.read()
    new_text = aggregate_text(raw_text)
    alphabet, reverse_alphabet = get_alphabets(new_text)
    logging.info(f'-> created alphabet with {len(alphabet.keys())} chars')
    dataset = get_dataset(new_text)
    logging.info(f'-> formed dataset with {len(dataset)} sentences')
    for data in dataset:
        for i in range(0, len(data) - MAXLEN, STEP):
            sentences.append(((data[i: i + MAXLEN]), (data[i + MAXLEN])))

