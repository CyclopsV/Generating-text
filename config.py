import logging


class Config:
    file: str = 'Wonderland_ru.txt'  # Название файла для обучения сети
    epochs: int = 10  # Количество эпох обучения
    len_segment: int = 100  # Размер сегмента текста для обучения
    len_chars: int = 500  # Количестов генерируемых символов

    # Настройки логера
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    format_log = logging.Formatter(fmt='%(levelname)s:%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(format_log)
    logger.addHandler(handler)
