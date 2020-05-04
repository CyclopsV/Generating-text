import logging


class Config:
    file: str = 'Wonderland_ru.txt'  # Название файла для обучения сети
    epochs: int = 1  # Количество эпох обучения
    len_segment: int = 150  # Размер сегмента текста для обучения
    len_predict_chars: int = 500  # Количестов генерируемых символов
    len_batch: int = 100  # Количество сегментво в пакете
    start_text: str = 'алиса'

    # Настройки логера
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    format_log = logging.Formatter(fmt='%(levelname)s:%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(format_log)
    logger.addHandler(handler)
