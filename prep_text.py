import re
from os.path import isfile
from typing import Tuple

from numpy.core.multiarray import ndarray, array

from config import Config

logger = Config.logger


def get_text(file_name: str) -> str:
    """Полуение текста из файла"""
    if isfile('clean_text.txt'):
        logger.info('Найден файл с чистым текстом')
        with open('clean_text.txt') as file_handler:
            raw_text: str = file_handler.read()
    else:
        logger.info('Отчистка текста')
        with open(file_name) as file_handler:
            raw_text: str = file_handler.read()
        raw_text = throw_trash(raw_text)
        logger.info('Создание файла с читсым текстом')
        with open('clean_text.txt', 'w') as file_handler:
            file_handler.write(raw_text)
    logger.info(f'Текст [:100]:\n    {raw_text[:100]}')
    return raw_text


def throw_trash(string: str) -> str:
    """Удаление из текста лишьних символов"""
    reg: re.Pattern = re.compile('[^а-яА-ЯЁё ]')
    string: str = reg.sub('', string)
    string = re.sub(' +', ' ', string)
    return string.lower()


def get_batches(coded_text: ndarray, batch_size: int = Config.len_batch, segment_len: int = Config.len_segment,
                s: str = '') -> Tuple[ndarray, ndarray]:
    """Нарезка текста на пакеты"""
    chars_in_batch: int = (segment_len + 1) * batch_size
    number_batches: int = int(len(coded_text) / chars_in_batch)
    index_last_segment: int = chars_in_batch * number_batches
    logger.info(
        f'Нарезка текста. ({s})\nДлина текста: {len(coded_text)}, Длина отрезков: {segment_len}, Количество отрезков в пакете: {batch_size}, Потеряно символов: {len(coded_text) - index_last_segment}')
    coded_text = coded_text[:index_last_segment]
    coded_text = coded_text.reshape((-1, batch_size, segment_len + 1))
    x: list = []
    y: list = []
    for i in coded_text:
        x_i = i[:, :segment_len]
        x.append(x_i)
        y_i = i[:, 1:]
        y.append(y_i)
    return array(x), array(y)
