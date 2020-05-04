from torch import load

from coding import count_chars, coding_chars, coding_text
from config import Config
from network import RNN
from prep_text import get_text, throw_trash

text = get_text(Config.file)
text = throw_trash(text)

count_c, chars = count_chars(text)
chars_to_int, int_to_chars = coding_chars(count_c)
coded_text = coding_text(text, chars_to_int)

network = RNN([int_to_chars, chars_to_int], hidden=512, layers=3)
network.train_net(coded_text)
network.predict(Config.start_text)

file_name = input('Введите названия файла состояний: ')
with open(f'weights/{file_name}.net', 'rb') as f:
    checkpoint = load(f)

network = RNN(checkpoint['coded_dicts'], hidden=checkpoint['hidden'], layers=checkpoint['layers'])
network.load_state_dict(checkpoint['state_dict'])
network.predict(Config.start_text)
