# stepik-dl-nlp
В этом репозитории представлены материалы мини-курса на Stepik "Нейронные сети и обработка текста" и мои ноутбуки с решениям теоретических задач (в папке theor_tasks) и задач с семинаров.


## Инструкция по запуску 

Чтобы запустить ноутбук с семинара на своем ноутбуке:

1) Cклонируйте репозиторий курса:

`git clone https://github.com/Samsung-IT-Academy/stepik-dl-nlp.git`

2) В терминале выполните команду:

`pip install -r requirements.txt`

3) Запустите ноутбук:

`ipython notebook`
 

Чтобы запустить ноутбук на [Google Colab](https://colab.research.google.com):

1) Скачайте ноутбук (вкладка Github, затем прописываете адрес репозитория.

2) Запустите ноутбук.

3) Чтобы выкачать на colab библиотеку dlnlputils, не забудьте выполнить команду в первой ячейке:

```
!git clone https://github.com/Samsung-IT-Academy/stepik-dl-nlp.git && pip install -r stepik-dl-nlp/requirements.txt
import sys; sys.path.append('./stepik-dl-nlp')
```

4) Не забудьте настроить `device='cpu'` или `device='cuda'`, а также выбрать подходящий Runtime в Google Colab (CPU/TPU/GPU).

А также следуйте комментариям касательно путей внутри ноутбуков.

Ноутбуки также работают и на Kaggle Notebooks.


## Датасеты

New York Times

https://github.com/nytimes/ingredient-phrase-tagger 

Copyright (c) 2016 The New York Times Company
Licensed under the Apache License, Version 2.0 
