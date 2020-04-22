from keras.callbacks import ModelCheckpoint

from config import Config
from fun import get_text, count_word, coding_words, fin_coding, create_nn, create_text

logger = Config.logger

text = get_text(Config.file)
word_to_int = count_word(text, True)
word_to_int = coding_words(word_to_int)
text_segments, coded_chars, text_segments_int = fin_coding(Config.len_segment, text, word_to_int)
model = create_nn(text_segments.shape[1], text_segments.shape[2], coded_chars.shape[1], )

filepath = 'weights-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
logger.info('Запуск обучения')
model.fit(text_segments, coded_chars, epochs=Config.epochs, batch_size=128, callbacks=callbacks_list)

logger.info('Начало генерации теста')
create_text(Config.len_chars, text_segments_int, word_to_int, model)
logger.info('Генерация текста закочена')
