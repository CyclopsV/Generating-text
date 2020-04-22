import re
from os import listdir
from os.path import isfile
from typing import Tuple

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from numpy import reshape, ndarray, argmax
from numpy.random import randint

from config import Config

logger = Config.logger


def throw_trash(string: str) -> str:
    reg: re.Pattern = re.compile('[^а-яА-ЯЁё ]')
    string: str = reg.sub('', string)
    string = re.sub(' +', ' ', string)
    return string.lower()


def count_word(string: str, chars: bool = False) -> dict:
    if chars:
        s_list: list = list(string)
        out: str = 'Всего символов:'
    else:
        s_list: list = string.split(' ')
        out: str = 'Всего слов:'
    logger.info(f'{out} {len(s_list)}')
    count: dict = {}
    for i in set(s_list):
        count.update({i: s_list.count(i)})
    count = dict(sorted(count.items(), key=lambda x: x[1], reverse=True))
    return count


def coding_words(count_w: dict, all: bool = True) -> dict:
    words: dict = {}
    for i, c in enumerate(count_w):
        if all:
            words.update({c: i})
        else:
            if count_w[c] > 1:
                words.update({c: i + 1})
            else:
                break
    logger.info(f'Уникальных символов: {len(words)}')
    logger.info(f'Кодировка: {words}')
    return words


def get_text(file_name: str) -> str:
    if isfile('clean_text.txt'):
        logger.info('Найден файл с чистым текстом')
        with open('clean_text.txt') as file_handler:
            raw_text: str = file_handler.read()
    else:
        logger.info('Чтение файла')
        with open(file_name) as file_handler:
            raw_text: str = file_handler.read()
        raw_text = throw_trash(raw_text)
        logger.info('Создание файла с читсым текстом')
        with open('clean_text.txt', 'w') as file_handler:
            file_handler.write(raw_text)
    return raw_text


def fin_coding(len_segments: int, text: str, code_dict: dict) -> Tuple[ndarray, ndarray, list]:
    coded_segments: list = []
    coded_chars: list = []
    for i in range(0, len(text) - len_segments):
        segment: str = text[i:i + len_segments]
        coded_segment: list = []
        for char in segment:
            coded_segment.append(code_dict[char])
        coded_segments.append(coded_segment)
        last_char: str = text[i + len_segments]
        coded_chars.append(code_dict[last_char])
    logger.info(f'Текст разделен на {len(coded_segments)} частей')
    np_coded_segments: ndarray = reshape(coded_segments, (len(coded_segments), len_segments, 1))
    np_coded_segments = np_coded_segments / float(len(code_dict))
    kr_coded_chars: ndarray = to_categorical(coded_chars)
    return np_coded_segments, kr_coded_chars, coded_segments


def create_nn(in_x: int, in_y: int, out: int, training: bool = True, weights_file: str = None):
    model: Sequential = Sequential()
    model.add(LSTM(256, input_shape=(in_x, in_y), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(out, activation='softmax'))
    if not training:
        if weights_file:
            file_name: str = weights_file
        else:
            weights: list = []
            for i in listdir():
                if '.hdf5' in i:
                    weights.append(i[0:-5].split('-'))
            weights = min(weights, key=lambda x: x[-1])
            file_name: str = '-'.join(weights) + '.hdf5'
        logger.info(f'Выбран файл с весами: "{file_name}"')
        model.load_weights(file_name)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def get_decoded_dict(coded_dict: dict) -> dict:
    encoded_dict: dict = {}
    for i, c in enumerate(coded_dict):
        encoded_dict.update({i: c})
    return encoded_dict


def decoding_string(string: list, coded_dict: dict) -> str:
    encoded_dict: dict = get_decoded_dict(coded_dict)
    text: list = []
    for char in string:
        text.append(encoded_dict[char])
    return ''.join(text)


def create_chars(text: list, coded_dict: dict, model: Sequential) -> int:
    np_text: ndarray = reshape(text, (1, len(text), 1))
    np_text = np_text / float(len(coded_dict))
    predict: ndarray = model.predict(np_text, verbose=0)
    best: int = argmax(predict)
    return best


def create_text(n: int, text_segments: list, coded_dict: dict, model: Sequential) -> None:
    start: int = randint(0, len(text_segments) - 1)
    text: list = text_segments[start]
    start_text: str = decoding_string(text, coded_dict)
    print(f'Начало: "{start_text}"')
    for i in range(n):
        char = create_chars(text, coded_dict, model)
        print(decoding_string([char], coded_dict), end='', flush=True)
        text.pop(0)
        text.append(char)
    print()
