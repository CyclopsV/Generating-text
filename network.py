from typing import Tuple

from numpy import mean, ndarray, array
from numpy.random.mtrand import choice
from torch import Tensor, from_numpy, save, softmax
from torch.nn import Module, LSTM, Dropout, Linear, CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from coding import text_to_ohe, one_hot_encoding
from config import Config
from prep_text import get_batches

logger = Config.logger


class RNN(Module):
    def __init__(self, coded_dicts: list, hidden: int = 256, layers: int = 2, drop: float = 0.5,
                 learning_rate: float = 0.001):
        super().__init__()
        self.drop: float = drop
        self.layers: int = layers
        self.hidden: int = hidden
        self.lr: float = learning_rate
        self.int2char: dict = coded_dicts[0]
        self.char2int: dict = coded_dicts[1]

        self.lstm: LSTM = LSTM(len(self.int2char), hidden, layers, dropout=drop, batch_first=True)  # Основные слои сети
        self.dropout: Dropout = Dropout(drop)  # Слой отсева
        self.fin: Linear = Linear(hidden, len(self.int2char))  # Выходной слой
        logger.info(f'Создана сеть\n{self}')

    def forward(self, inputs: Tensor, hidden: tuple) -> Tuple[Tensor, tuple]:
        """Проход через сеть"""
        output: Tensor
        output, hidden = self.lstm(inputs.float(), hidden)
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.hidden)
        output = self.fin(output)
        logger.debug('Выполнен проход по сети')
        return output, hidden

    def init_hidden(self, batch_size: int) -> tuple:
        """Инициализация состояний скрытых слоев"""
        weight: Tensor = next(self.parameters()).data
        hidden: tuple = (weight.new(self.layers, batch_size, self.hidden).zero_(),
                         weight.new(self.layers, batch_size, self.hidden).zero_())
        logger.debug('Инициализированны скрытые состояния')
        return hidden

    def train_net(self, data: ndarray, epochs: int = Config.epochs, batch_size: int = Config.len_batch,
                  seq_length: int = Config.len_segment, learning_rate: float = 0.001, limit_grad: int = 5,
                  piece_text: float = 0.1, check_step: int = 10, first_epoch: int = 0):
        """ Тренеровка NN"""

        criterion: CrossEntropyLoss = CrossEntropyLoss()
        test_index: int = int(len(data) * (1 - piece_text))

        min_test_index: int = (seq_length + 1) * batch_size
        data_test: ndarray
        if len(data) - test_index < min_test_index:
            intersection: int = test_index - (len(data) - min_test_index)
            logger.info(f'Текста не достаточно для создания отдельных пакетов для обучения и проверки. '
                        f'Совпадают {intersection} символов')
            data_test = data[-min_test_index:]
        else:
            data_test = data[test_index:]

        data = data[:test_index]
        step: int = 0
        number_unique_chars: int = len(self.int2char)
        inputs_data, goals = get_batches(data, batch_size, seq_length, s='для обучения')
        inputs_data_test, goals_test = get_batches(data_test, batch_size, seq_length, s='для проверки')
        logger.info('Начало обучения')
        self.train()
        optimization: Adam = Adam(self.parameters(), lr=learning_rate)

        for epoch in range(first_epoch, first_epoch + epochs):
            hidden_states = self.init_hidden(batch_size)

            for input_data, goal in zip(inputs_data, goals):
                input_data = text_to_ohe(input_data, number_unique_chars)
                inputs: Tensor = from_numpy(input_data)
                targets: Tensor = from_numpy(goal)
                hidden_states = tuple([each.data for each in hidden_states])
                self.zero_grad()
                output, hidden_states = self(inputs.float(), hidden_states)
                loss: Tensor = criterion(output, targets.view(batch_size * seq_length).long())
                loss.backward()
                clip_grad_norm_(self.parameters(), limit_grad)
                optimization.step()

                if step % check_step == 0:
                    hidden_states_test = self.init_hidden(batch_size)
                    hidden_states_test = tuple([each.data for each in hidden_states_test])
                    losses_test: list = []
                    self.eval()

                    for input_data_test, goal_test in zip(inputs_data_test, goals_test):
                        input_data_test: ndarray = text_to_ohe(input_data_test, number_unique_chars)
                        input_data_test: Tensor = from_numpy(input_data_test)
                        goal_test: Tensor = from_numpy(goal_test)

                        hidden_states_test = tuple([each.data for each in hidden_states_test])
                        output, hidden_states_test = self(input_data_test.float(), hidden_states_test)
                        loss_test: Tensor = criterion(output, goal_test.view(batch_size * seq_length).long())
                        losses_test.append(loss_test.item())

                    self.train()

                step += 1

            loss_print: float = round(loss.item(), 4)
            losses_test_print: float = round(mean(losses_test), 4)
            logger.warning(
                f'Эпоха: {epoch + 1}/{epochs}   |   Потери: {loss_print}   |   Потери проверки: {losses_test_print}')
            checkpoint_file = f'weights/{epoch + 1}_{loss_print}_{losses_test_print}.net'
            checkpoint = {'hidden': self.hidden,
                          'layers': self.layers,
                          'state_dict': self.state_dict(),
                          'coded_dicts': [self.int2char, self.char2int]}
            with open(checkpoint_file, 'wb') as f:
                save(checkpoint, f)

    def predict_char(self, char: str, top: int = 5, hidden_states: tuple = None) -> Tuple[str, tuple]:
        """Предсказание символа"""
        x: ndarray = one_hot_encoding([self.char2int[char]], len(self.char2int))
        x = array([x])
        inputs: Tensor = from_numpy(x)
        hidden_states: tuple = tuple([hidden_state.data for hidden_state in hidden_states])
        out, hidden_states = self(inputs, hidden_states)
        probabilities: Tensor = softmax(out, dim=1).data
        top_chars: Tensor
        probabilities, top_chars = probabilities.topk(top)
        top_chars: ndarray = top_chars.numpy().squeeze()
        probabilities = probabilities.numpy().squeeze()
        char: int = choice(top_chars, p=probabilities / probabilities.sum())
        return self.int2char[char], hidden_states

    def predict(self, start_string: str, size: int = Config.len_predict_chars, top: int = 5):
        logger.info(f'Генерация текста ({size} символов)\nТекст для инициализации: {start_string}\nРезультат:')
        self.cpu()
        self.eval()
        # Предсказание символа следующего за начальными
        chars: list = [char for char in start_string]
        hidden_states = self.init_hidden(1)

        for char_start in start_string:
            char, hidden_states = self.predict_char(char_start, top, hidden_states)

        print(start_string, end='-> ')
        print(char, end='')
        chars.append(char)

        # Предсказание последующих символов
        for i in range(size):
            char, hidden_states = self.predict_char(chars[-1], top, hidden_states)
            chars.append(char)
            print(char, end='')

        print()
