conversations = []                                # Заготовим список для пар фраз

with open("base/rus.txt", 'r', encoding='utf-8') as f: # Открываем файл словаря в режиме чтения
    lines = f.read().split('\n')                  # Читаем весь файл, режем на строки

# Цикл по строкам
for i,line in enumerate(lines):
    if i>50000:                                         # Нам нужно только 50000 первых строк
        break     
    try:
        input_text, target_text,_ = line.split("\t")    # Берем очередную строку, режем по символу табуляции
        conversations.append([input_text, target_text]) # Заполняем список пар фраз
    except:
        continue     

from replacer import my_replacer

# Собираем вопросы и ответы в списки

questions = [] # Переменная для списка входных фраз
answers = []   # Переменная для списка ответных фраз

# Цикл по всем парам фраз
for con in conversations:

    if len(con) > 1 :                       # Если ответная фраза содержит более одно двух предложений
        questions.append(my_replacer(con[0])) # То первую в списке фразу отправляем в список входных фраз
        replies = my_replacer(con[1:])        # А ответную составляем из последующих строк
        ans = ' '.join(replies)               # Здесь соберем ответ
        answers.append(ans)                   # Добавим в список ответов
    else:
        continue                              # Иначе идем на новой парой фраз

# Добавим в каждую ответную фразу теги  <START> и <END>
answers = ['<START> ' + s + ' <END>' for s in answers]
