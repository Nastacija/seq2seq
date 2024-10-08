from tensorflow.keras.preprocessing.sequence import pad_sequences
from replacer import my_replacer
from tokenizer import tokenizer, maxLenQuestions

def strToTokens(sentence: str):

    ''' Функция для удаления пробелов перед знаками препинания

        Args: фраза

        Returns: список токенов
    '''

    # Почистим фразу
    tmp_sent = my_replacer(sentence)

    # Приведем предложение к нижнему регистру и разбирает на слова
    words = tmp_sent.lower().split()

    # Создадим список для последовательности токенов/индексов
    tokensList = list()

    # Для каждого слова в предложении
    for word in words:

        try:
            tokensList.append(tokenizer.word_index[word]) # Определяем токенайзером индекс и добавляем в список
        except:
            pass # Слова нет - просто игнорируем его

    # Вернёт входную фразу в виде последовательности индексов
    if tokensList:
        return pad_sequences([tokensList], maxlen=maxLenQuestions , padding='post')

    # Фраза из незнакомых слов - вернем None
    return None