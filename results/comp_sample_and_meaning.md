# Постановка задачи

Есть список слов-омонимов.
Для каждого слова есть 2 и более определения.
Для каждого слова есть набор размеченных текстов, где это слово встречается в одном из смыслов, упомянутых выше.
Требуется автоматически выбрать одно определение омонима из списка определений так, чтобы оно совпало с тем, что указанно в разметке.

Пример размеченного омонима:
```json
"байка": {
        "meanings": [
            {
                "index": 0,
                "определение": "юмористический рассказ, как правило, основанный на реальных событиях"
            },
            {
                "index": 1,
                "определение": "мягкая ворсистая хлопчатобумажная или шерстяная, полушерстяная ткань, одежда из такой ткани"
            }
        ],
        "samples": [
            {
                "text": "А пока Вы читали мои байки, я вспомнила ещё одну историю",
                "meaning": 0
            },
            {
                "text": "Фланель и байка имеют на поверхности маленькие ворсинки, из них тоже можно шить зверушек с мягкой шкуркой",
                "meaning": 1
            },
            {
                "text": "она всё лето прела под чёрной байкой для того, чтобы иметь удовольствие показывать шлейф чрезмерной длины",
                "meaning": 1
            }
        ]
    },
```

Корневой элемент - омоним, который содержит два списка: список определений и список примеров на каждое определение.

Для решения задачи использовалось 2 подхода:
1. Подход со сравнением примера и значения
2. Подход с обучением отдельного классификатора для каждого омонима

## Подход со сравнением примера и значения

### Описание

Определения и примеры переводились в вектора, потом каждый пример сравнивался с каждым определением. По наибольшему показателю метрики принималось решение, что пример соответствует конкретному определению.

Для преобразования текстов в вектора использовались различные эмбеддинги: w2v, d2v, navec, gensim_pretrainde, bert различных моделей: cointegrated/rubert-tiny, cointegrated/rubert-tiny2, sberbank-ai/sbert_large_nlu_ru, sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, DeepPavlov/rubert-base-cased-sentence, DeepPavlov/rubert-base-cased, inkoziev/sbert_synonymy

Тексты преобразовывались в вектора с варьированием лемматизации и удаления слов.
Итоговый вектор текста получался тремя способами: либо вычислялось среднеарифметическое значение вектора поэлементно и получался вектор из среднеарифметических значений, либо значения векторов складывались поэлементно, либо использовался как есть, если его давали на весь текст, а не на каждое слово отдельно.

Метрики: косинусное сходство, метрики из sklearn: euclidean, manhattan, minkowski, hamming, canberra, braycurtis.

Корпуса:
* homonyms_ru_clean.json - корпус вначале собранный с сайта homonyms.ru и потом вручную очищен.
* homonyms_ru_dirty.json - корпус собранный с сайта homonyms.ru без ручной обработки.
* homonyms_with_50_samples.json - корпус собранный с различных сайтов вручную. Каждый омоним имеет минимум по 20 примеров и не менее 50 примеров в общем.


### Выводы

После очистки корпуса homonyms_ru он стал вдвое меньше и результаты по нему выросли на 10% и более процентов, в зависимости от метода.

Лучше всего с задачей снятия омонимов справляется эмбеддинг на основе bert с моделями rubert-tiny2 и paraphrase-multilingual-MiniLM-L12-v2. Лучший результат находится на уровне 0.70 доли правильно снятых омонимов. В них разница между корпусами homonyms_with_50_samples и homonyms_ru_clean минимальна, не более двух процентов.
Ввиду того, что первый корпус собирался полностью вручную и имеет в 2 раза меньше различных омонимов для снятия, считаю, что метод справляется достаточно хорошо, так как разница в результате небольшая.


### Метод w2v

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | manhattan | 10 | 279/586 | 0.4761 | лемматизация = False, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_ru_clean.json | manhattan | 26 | 103/187 | 0.5508 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | canberra | 57 | 154/414 | 0.3720 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |


### Метод navec

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 348/586 | 0.5939 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | similarity_cosine | 26 | 98/187 | 0.5241 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 188/419 | 0.4487 | лемматизация = True, Удаление стоп-слов = True |

### Метод gensim_pretrainde, модель = word2vec-ruscorpora-300

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | braycurtis | 10 | 380/586 | 0.6485 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_clean.json | canberra | 26 | 113/187 | 0.6043 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | canberra | 57 | 216/419 | 0.5155 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |

### Метод d2v

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 266/586 | 0.4539 | лемматизация = False, Удаление стоп-слов = True |
| homonyms_ru_dirty.json | euclidean | 57 | 154/420 | 0.3667 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | canberra | 26 | 94/187 | 0.5027 | лемматизация = True, Удаление стоп-слов = False |

### Метод bert_score, модель: cointegrated/rubert-tiny

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | canberra | 10 | 343/586 | 0.5853 |
| homonyms_ru_clean.json | canberra | 26 | 107/187 | 0.5722 |
| homonyms_ru_dirty.json | canberra | 57 | 193/420 | 0.4595 |


### Метод bert_score, модель: cointegrated/rubert-tiny2

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | braycurtis | 10 | 414/586 | 0.7065 |
| homonyms_ru_clean.json | manhattan | 26 | 129/187 | 0.6898 |
| homonyms_ru_dirty.json | manhattan | 57 | 230/420 | 0.5476 |


### Метод bert_score, модель: sberbank-ai/sbert_large_nlu_ru

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | minkowski | 10 | 387/586 | 0.6604 |
| homonyms_ru_clean.json | manhattan | 26 | 117/187 | 0.6257 |
| homonyms_ru_dirty.json | manhattan | 57 | 208/420 | 0.4952 |


### Метод bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | braycurtis | 10 | 416/586 | 0.7099 |
| homonyms_ru_clean.json | braycurtis | 26 | 131/187 | 0.7005 |
| homonyms_ru_dirty.json | braycurtis | 57 | 241/420 | 0.5738 |


### Метод bert_score, модель: DeepPavlov/rubert-base-cased-sentence

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | manhattan | 10 | 392/586 | 0.6689 |
| homonyms_ru_dirty.json | minkowski | 57 | 227/420 | 0.5405 |
| homonyms_ru_clean.json | braycurtis | 26 | 119/187 | 0.6364 |


### Метод bert_score, модель: DeepPavlov/rubert-base-cased

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | canberra | 10 | 301/586 | 0.5137 |
| homonyms_ru_dirty.json | canberra | 57 | 162/420 | 0.3857 |
| homonyms_ru_clean.json | braycurtis | 26 | 107/187 | 0.5722 |


### Метод bert_score, модель: inkoziev/sbert_synonymy

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | canberra | 10 | 333/586 | 0.5683 |
| homonyms_ru_dirty.json | manhattan | 57 | 175/420 | 0.4167 |
| homonyms_ru_clean.json | canberra | 26 | 113/187 | 0.6043 |


## Подход с обученим отдельного классификатора для каждого омонима

### Описание

Примеры переводились в вектора, каждому из которых ставилась в соответствие метка - номер определения.
Далее 60% примеров бралось на обучение и 40% на тест.
Для классификации использовались классификаторы svm, RandomForest, K Neighbors с различными параметрами.

Для преобразования текстов в вектора использовались различные эммбединги: w2v, tfidf, bert различных моделей: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, cointegrated/rubert-tiny2, cointegrated/rubert-tiny.

Корпуса:
* homonyms_ru_clean.json - корпус вначале собранный с сайта homonyms.ru и потом вручную очищен.
* homonyms_with_50_samples.json - корпус собранный с различных сайтов вручную. Каждый омоним имеет минимум по 20 примеров и не менее 50 примеров в общем.

С корпусом homonyms_ru_dirty.json классификаторы не завелись из-за нехватки данных для разметки.

### Выводы

Здесь также, как и в предыдущем разделе лучше всего справились bert-модели paraphrase-multilingual-MiniLM-L12-v2 и rubert-tiny2 с результатом 0.8359 - 0.8509 средних микро- и макро-f-меры на корпусе homonyms_with_50_samples. У второго корпуса результаты отстают на 30-40%.
Это можно объяснить тем, что корпус homonyms_with_50_samples имеет больше примеров на каждый омоним, что позитивно влияет на обучение. Кроме того, корпус homonyms_ru_clean имеет больше разлиных омонимов с меньшим количеством примеров, что сильно сказывается на показателях метрик.


### Метод w2v

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | RandomForest | 0.3704 |0.4392 | n_estimators=150 entropy bootstrap=False |
| homonyms_with_50_samples.json | RandomForest |  0.5608 | 0.5762 | n_estimators=150 entropy bootstrap=False |


### Метод Tfidf

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | SVM | 0.6999 | 0.7039 | class_weight=balanced linear |
| homonyms_ru_clean.json | RandomForest | 0.4332  | 0.3237 | entropy max_features=log2 |


### Метод bert_score, модель: rubert-tiny

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | SVM | 0.7686 | 0.7743 | class_weight=balanced kernel=poly coef0=0.75 |
| homonyms_ru_clean.json | SVM | 0.4330 | 0.5165 | class_weight=balanced linear | 


### Метод bert_score, модель: paraphrase-multilingual-MiniLM-L12-v2

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | SVM | 0.8464 | 0.8509 | kernel=poly degree=4 coef0=0.75  |
| homonyms_ru_clean.json | RandomForest | 0.3171 | 0.4219 | n_estimators=150 entropy bootstrap=False |

### Метод bert_score, модель: rubert-tiny2

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | SVM | 0.8359 | 0.8401 | class_weight=balanced kernel=poly coef0=0.75 |
| homonyms_ru_clean.json | SVM |  0.4158 | 0.5088 | class_weight=balanced linear |