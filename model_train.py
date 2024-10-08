# Подгрузим модели кераса
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
model.compile(optimizer=RMSprop(), loss='sparse_categorical_crossentropy')

from tokenizer import encoderForInput, decoderForInput, decoderForOutput

# Запустим обучение
model.fit([encoderForInput , decoderForInput], decoderForOutput, batch_size=128, epochs=5)
model.save_weights(r'model_checkpoint.weights.h5')


