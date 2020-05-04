from typing import Tuple

from numpy.core.multiarray import ndarray, array, zeros

from config import Config

logger = Config.logger


def count_chars(string: str) -> Tuple[dict, list]:
    """Подсчет символов"""
    s_list: list = list(string)
    chars: list = list(set(s_list))
    logger.info(f'Уникальных символов: {len(chars)}')
    logger.info(f'Всего символов: {len(s_list)}')
    count: dict = {}
    for i in chars:
        count.update({i: s_list.count(i)})
    count = dict(sorted(count.items(), key=lambda x: x[1], reverse=True))
    return count, chars


def coding_chars(count_c: dict) -> Tuple[dict, dict]:
    """Кодировка символов"""
    chars: dict = {}
    chars_rev: dict = {}
    for i, c in enumerate(count_c):
        chars.update({c: i})
        chars_rev.update({i: c})
    logger.info(f'Кодировка\n    char->int: {chars}\n    int->char: {chars_rev}')
    return chars, chars_rev


def coding_text(text: str, coded_dict: dict) -> ndarray:
    """Кодирование текста"""
    coded_text: list = []
    for char in text:
        coded_text.append(coded_dict[char])
    logger.info(f'Закодированный текст [:100]:\n    {coded_text[:100]}')
    return array(coded_text)


def one_hot_encoding(chars_list: list, number_chars: int) -> ndarray:
    """One-hot encoding (Унитарное кодирование) символов"""
    chars: ndarray = zeros((len(chars_list), number_chars))
    for i in range(len(chars)):
        chars[i, chars_list[i]] = 1
    return chars


def text_to_ohe(batch: ndarray, coded_dict: int) -> ndarray:
    """Кодирование текста -> One-hot"""
    coded_batch: list = []
    for segment in batch:
        segment = one_hot_encoding(list(segment), coded_dict)
        coded_batch.append(segment)
    return array(coded_batch)
