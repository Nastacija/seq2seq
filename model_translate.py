#Подгрузим модели кераса
from tensorflow.keras.models import Model
# Подключим нужные слои
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
# Поключим оптимайзеры
from tensorflow.keras.optimizers import RMSprop

from tokenizer import vocabularySize

# Создадим энкодер
encoderInputs = Input(shape=(None , ))                                             # Добавим входной слой
encoderEmbedding = Embedding(vocabularySize, 200 , mask_zero=True)(encoderInputs)  # Добавим эмбеддинг
encoderOutputs, state_h , state_c = LSTM(200, return_state=True)(encoderEmbedding) # Добавим LSTM
encoderStates = [state_h, state_c]                                                 # Соберем выходы lstm  в список 

# Создадим декодер
decoderInputs = Input(shape=(None, ))                                                # Добавим входной слой
decoderEmbedding = Embedding(vocabularySize, 200, mask_zero=True) (decoderInputs)    # Добавим эмбеддинг
decoderLSTM = LSTM(200, return_state=True, return_sequences=True)                    # Создадим LSTM слой
decoderOutputs , _ , _ = decoderLSTM (decoderEmbedding, initial_state=encoderStates) # Прогоним выход embedding через LSTM
decoderDense = Dense(vocabularySize, activation='softmax')                           # Создадим dense слой
output = decoderDense (decoderOutputs)                                               # Прогоним  выход LSTM через DENSE

model = Model([encoderInputs, decoderInputs], output)
model.load_weights(r'model_checkpoint.weights.h5')

# Создадим модель кодера, на входе далее будут закодированные вопросы, на выходе состояния state_h, state_c
encoderModel = Model(encoderInputs, encoderStates)
# Создадим модель декодера
decoderStateInput_h = Input(shape=(200 ,)) # Добавим входной слой для state_h
decoderStateInput_c = Input(shape=(200 ,)) # Добавим входной слой для state_c
# Соберем оба inputs вместе и запишем в decoderStatesInputs
decoderStatesInputs = [decoderStateInput_h, decoderStateInput_c]
# Берём ответы, прошедшие через эмбединг, вместе с состояниями и подаём LSTM cлою
decoderOutputs, state_h, state_c = decoderLSTM(decoderEmbedding, initial_state=decoderStatesInputs)

# LSTM даст нам новые состояния
decoderStates = [state_h, state_c]

# И ответы, которые мы пропустим через полносвязный слой с софтмаксом
decoderOutputs = decoderDense(decoderOutputs)
# Определим модель декодера, на входе далее будут раскодированные ответы (decoderForInputs) и состояния
# на выходе предсказываемый ответ и новые состояния
decoderModel = Model([decoderInputs] + decoderStatesInputs, [decoderOutputs] + decoderStates)

from str2tokens import strToTokens
from tokenizer import tokenizer, maxLenAnswers
import numpy as np

for _ in range(2):

    # подготовка

    qua  = strToTokens(input('Исходное предложение на английском: '))
    if qua is None:
        print ("а вот спросите меня о чем-нить полезном: ")  # Выдадим дежурную фразу
        continue                                             # Пойдем за следущей фразой

    emptyTargetSeq = np.zeros((1, 1))
    emptyTargetSeq[0, 0] = tokenizer.word_index['start']
    stopCondition = False
    decodedTranslation = ''
    statesValues = encoderModel.predict(qua)

    # пока не сработало стоп-условие
    while not stopCondition:

        # В модель декодера подадим пустую последовательность со словом 'start' и состояния
        decOutputs , h , c = decoderModel.predict([emptyTargetSeq] + statesValues)
        # Получим индекс предсказанного слова.
        sampledWordIndex = np.argmax( decOutputs[0, 0, :])
        # Создаем переменную для преобразованных на естественный язык слов
        sampledWord = None

        # Переберем в цикле все индексы токенайзера
        for word, index in tokenizer.word_index.items():

            # Если индекс выбранного слова соответствует какому-то индексу из словаря
            if sampledWordIndex == index:
                # Слово, идущее под этим индексом в словаре, добавляется в итоговый ответ
                decodedTranslation += ' {}'.format(word)
                # Выбранное слово фиксируем в переменную sampledWord
                sampledWord = word

        # Если выбранным словом оказывается 'end' либо если сгенерированный ответ превышает заданную максимальную длину ответа
        if sampledWord == 'end' or len(decodedTranslation.split()) > maxLenAnswers:
            stopCondition = True # Срабатывает стоп-условие и прекращаем генерацию

        # Создаем пустой массив
        emptyTargetSeq = np.zeros((1, 1))

        # Заносим в него индекс выбранного слова
        emptyTargetSeq[0, 0] = sampledWordIndex

        # Записываем состояния, обновленные декодером
        statesValues = [h, c]

        # И продолжаем цикл с обновленными параметрами

    # Выводим ответ сгенерированный декодером
    print("Перевод: ", decodedTranslation)