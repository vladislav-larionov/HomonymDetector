# /home/vladislav/projects/python/HomonymDetector/comp_sample_and_meaning/compare_sample_and_meaning.py

## Описание

Корпус структурирован следующим образом: 
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
Корневой элемент - омоним, который содежит два списка: список определений и список примеров на каждое определение.
Определения и примеры переводились в вектора, потом каждый пример сравнивался с каждым определением. По наибольшему показателю метрики принималось решение, что пример соответствует конкретному определению.

Для преобразования текстов в вектора использовались различные эммбединги: w2v, d2v, navec, gensim_pretrainde, bert различных моделей: cointegrated/rubert-tiny, cointegrated/rubert-tiny2, sberbank-ai/sbert_large_nlu_ru, sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, DeepPavlov/rubert-base-cased-sentence, DeepPavlov/rubert-base-cased, inkoziev/sbert_synonymy

Тексты преобразовывались в вектора с варьированием леммантизации и удаления слов.
Итоговый вектор текста получался тремя способами: либо вычислялось среднеарифметическое значение вектора поэлементно и получался вектор из среднеарифметических значений, либо значения векторов складывались поэлементно, либо использовался как есть, если его давали на весь текст, а не на каждое слово отдельно.

Метрики: косинусное сходство, метрики из sklearn: euclidean, manhattan, minkowski, hamming, canberra, braycurtis.

Корпуса:
* homonyms_ru_clean.json - корпус вначале собранный с сайта homonyms.ru и потом вручную очищенн.
* homonyms_ru_dirty.json - корпус собранный с сайта homonyms.ru без ручной обработки.
* homonyms_with_50_samples.json - корпус собранный с различных сайтов вручную. Каждый омоним имеет минимум по 20 примеров и не менее 50 примеров в общем.

## Метод w2v

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 81/187 | 0.4332 | лемматизация = False, Удаление стоп-слов = True, Вектор - среднеарифметическое значение поэлемент
| homonyms_ru_dirty.json | similarity_cosine | 57 | 140/419 | 0.3341 | лемматизация = True, Удаление стоп-слов = True, Вектор - среднеарифметическое значение поэлементно |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 261/586 | 0.4454 | лемматизация = False, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_clean.json | euclidean | 26 | 97/187 | 0.5187 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | euclidean | 57 | 155/419 | 0.3699 | лемматизация = False, Удаление стоп-слов = True, Вектор - среднеарифметическое значение поэлементно |
| homonyms_with_50_samples.json | euclidean | 10 | 273/586 | 0.4659 | лемматизация = False, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_ru_clean.json | manhattan | 26 | 103/187 | 0.5508 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | manhattan | 57 | 153/419 | 0.3652 | лемматизация = False, Удаление стоп-слов = True, Вектор - среднеарифметическое значение поэлементно |
| homonyms_with_50_samples.json | manhattan | 10 | 279/586 | 0.4761 | лемматизация = False, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_ru_clean.json | minkowski | 26 | 97/187 | 0.5187 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | minkowski | 57 | 155/419 | 0.3699 | лемматизация = False, Удаление стоп-слов = True, Вектор - среднеарифметическое значение поэлементно |
| homonyms_with_50_samples.json | minkowski | 10 | 273/586 | 0.4659 | лемматизация = False, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_ru_clean.json | hamming | 26 | 96/187 | 0.5134 | лемматизация = False, Удаление стоп-слов = False, Вектор - сумма значений поэлементно |
| homonyms_ru_dirty.json | hamming | 57 | 151/419 | 0.3604 | лемматизация = False, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 | лемматизация = False, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_clean.json | canberra | 26 | 91/187 | 0.4866 | лемматизация = False, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | canberra | 57 | 154/414 | 0.3720 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_with_50_samples.json | canberra | 10 | 264/586 | 0.4505 | лемматизация = False, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_ru_clean.json | braycurtis | 26 | 94/187 | 0.5027 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | braycurtis | 57 | 148/414 | 0.3575 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_with_50_samples.json | braycurtis | 10 | 274/586 | 0.4676 | лемматизация = False, Удаление стоп-слов = True, Вектор - сумма значений поэлементно 




## Метод navec

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 98/187 | 0.5241 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 188/419 | 0.4487 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 348/586 | 0.5939 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | euclidean | 26 | 98/187 | 0.5241 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | euclidean | 57 | 188/419 | 0.4487 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | euclidean | 10 | 348/586 | 0.5939 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | manhattan | 26 | 98/187 | 0.5241 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | manhattan | 57 | 188/419 | 0.4487 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | manhattan | 10 | 348/586 | 0.5939 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | minkowski | 26 | 98/187 | 0.5241 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | minkowski | 57 | 188/419 | 0.4487 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | minkowski | 10 | 348/586 | 0.5939 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | hamming | 26 | 98/187 | 0.5241 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | hamming | 57 | 188/419 | 0.4487 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | hamming | 10 | 348/586 | 0.5939 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | canberra | 26 | 98/187 | 0.5241 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | canberra | 57 | 188/419 | 0.4487 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | canberra | 10 | 348/586 | 0.5939 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | braycurtis | 26 | 98/187 | 0.5241 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | braycurtis | 57 | 188/419 | 0.4487 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | braycurtis | 10 | 348/586 | 0.5939 | лемматизация = True, Удаление стоп-слов = True |

## Метод gensim_pretrainde, модель = word2vec-ruscorpora-300

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 73/187 | 0.3904 | лемматизация = False, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 129/419 | 0.3079 | лемматизация = False, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 227/586 | 0.3874 | лемматизация = False, Удаление стоп-слов = True, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_clean.json | euclidean | 26 | 105/186 | 0.5645 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_ru_dirty.json | euclidean | 57 | 195/413 | 0.4722 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_with_50_samples.json | euclidean | 10 | 344/586 | 0.5870 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_clean.json | manhattan | 26 | 104/186 | 0.5591 | лемматизация = True, Удаление стоп-слов = False, Вектор - сумма значений поэлементно |
| homonyms_ru_dirty.json | manhattan | 57 | 192/413 | 0.4649 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_with_50_samples.json | manhattan | 10 | 341/586 | 0.5819 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_ru_clean.json | minkowski | 26 | 105/186 | 0.5645 | лемматизация = True, Удаление стоп-слов = False, Вектор - сумма значений поэлементно |
| homonyms_ru_dirty.json | minkowski | 57 | 195/413 | 0.4722 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_with_50_samples.json | minkowski | 10 | 344/586 | 0.5870 | лемматизация = True, Удаление стоп-слов = True, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_clean.json | hamming | 26 | 88/186 | 0.4731 | лемматизация = True, Удаление стоп-слов = True, Вектор - сумма значений поэлементно |
| homonyms_ru_dirty.json | hamming | 57 | 153/413 | 0.3705 | лемматизация = True, Удаление стоп-слов = False, Вектор - сумма значений поэлементно |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_clean.json | canberra | 26 | 113/187 | 0.6043 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | canberra | 57 | 216/419 | 0.5155 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_with_50_samples.json | canberra | 10 | 378/586 | 0.6451 | лемматизация = True, Удаление стоп-слов = True, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_clean.json | braycurtis | 26 | 109/187 | 0.5829 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_ru_dirty.json | braycurtis | 57 | 212/419 | 0.5060 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| homonyms_with_50_samples.json | braycurtis | 10 | 380/586 | 0.6485 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |

## Метод d2v_emb

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 85/187 | 0.4545 | лемматизация = False, Удаление стоп-слов = True |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 144/420 | 0.3429 | лемматизация = False, Удаление стоп-слов = False |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 266/586 | 0.4539 | лемматизация = False, Удаление стоп-слов = True |
| homonyms_ru_clean.json | euclidean | 26 | 91/187 | 0.4866 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | euclidean | 57 | 154/420 | 0.3667 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | euclidean | 10 | 257/586 | 0.4386 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | manhattan | 26 | 91/187 | 0.4866 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | manhattan | 57 | 144/420 | 0.3429 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | manhattan | 10 | 254/586 | 0.4334 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | minkowski | 26 | 91/187 | 0.4866 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | minkowski | 57 | 153/420 | 0.3643 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_with_50_samples.json | minkowski | 10 | 257/586 | 0.4386 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | hamming | 26 | 88/187 | 0.4706 | лемматизация = False, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | hamming | 57 | 151/420 | 0.3595 | лемматизация = False, Удаление стоп-слов = False |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 | лемматизация = False, Удаление стоп-слов = False |
| homonyms_ru_clean.json | canberra | 26 | 94/187 | 0.5027 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | canberra | 57 | 151/420 | 0.3595 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_with_50_samples.json | canberra | 10 | 258/586 | 0.4403 | лемматизация = True, Удаление стоп-слов = True |
| homonyms_ru_clean.json | braycurtis | 26 | 93/187 | 0.4973 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_ru_dirty.json | braycurtis | 57 | 140/420 | 0.3333 | лемматизация = True, Удаление стоп-слов = False |
| homonyms_with_50_samples.json | braycurtis | 10 | 256/586 | 0.4369 | лемматизация = True, Удаление стоп-слов = True |

## Метод bert_score, модель: cointegrated/rubert-tiny

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 54/187 | 0.2888 |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 108/420 | 0.2571 |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 200/586 | 0.3413 |
| homonyms_ru_clean.json | euclidean | 26 | 101/187 | 0.5401 |
| homonyms_ru_dirty.json | euclidean | 57 | 185/420 | 0.4405 |
| homonyms_with_50_samples.json | euclidean | 10 | 327/586 | 0.5580 |
| homonyms_ru_clean.json | manhattan | 26 | 102/187 | 0.5455 |
| homonyms_ru_dirty.json | manhattan | 57 | 183/420 | 0.4357 |
| homonyms_with_50_samples.json | manhattan | 10 | 329/586 | 0.5614 |
| homonyms_ru_clean.json | minkowski | 26 | 101/187 | 0.5401 |
| homonyms_ru_dirty.json | minkowski | 57 | 185/420 | 0.4405 |
| homonyms_with_50_samples.json | minkowski | 10 | 327/586 | 0.5580 |
| homonyms_ru_clean.json | hamming | 26 | 88/187 | 0.4706 |
| homonyms_ru_dirty.json | hamming | 57 | 151/420 | 0.3595 |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 |
| homonyms_ru_clean.json | canberra | 26 | 107/187 | 0.5722 |
| homonyms_ru_dirty.json | canberra | 57 | 193/420 | 0.4595 |
| homonyms_with_50_samples.json | canberra | 10 | 343/586 | 0.5853 |
| homonyms_ru_clean.json | braycurtis | 26 | 100/187 | 0.5348 |
| homonyms_ru_dirty.json | braycurtis | 57 | 186/420 | 0.4429 |
| homonyms_with_50_samples.json | braycurtis | 10 | 330/586 | 0.5631 |


## Метод bert_score, модель: cointegrated/rubert-tiny2

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 32/187 | 0.1711 |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 84/420 | 0.2000 |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 128/586 | 0.2184 |
| homonyms_ru_clean.json | euclidean | 26 | 128/187 | 0.6845 |
| homonyms_with_50_samples.json | euclidean | 10 | 396/586 | 0.6758 |
| homonyms_ru_dirty.json | euclidean | 57 | 219/420 | 0.5214 |
| homonyms_ru_clean.json | manhattan | 26 | 129/187 | 0.6898 |
| homonyms_ru_dirty.json | manhattan | 57 | 230/420 | 0.5476 |
| homonyms_with_50_samples.json | manhattan | 10 | 404/586 | 0.6894 |
| homonyms_ru_clean.json | minkowski | 26 | 128/187 | 0.6845 |
| homonyms_ru_dirty.json | minkowski | 57 | 219/420 | 0.5214 |
| homonyms_with_50_samples.json | minkowski | 10 | 396/586 | 0.6758 |
| homonyms_ru_clean.json | hamming | 26 | 88/187 | 0.4706 |
| homonyms_ru_dirty.json | hamming | 57 | 151/420 | 0.3595 |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 |
| homonyms_ru_clean.json | canberra | 26 | 120/187 | 0.6417 |
| homonyms_ru_dirty.json | canberra | 57 | 213/420 | 0.5071 |
| homonyms_with_50_samples.json | canberra | 10 | 413/586 | 0.7048 |
| homonyms_ru_clean.json | braycurtis | 26 | 127/187 | 0.6791 |
| homonyms_ru_dirty.json | braycurtis | 57 | 222/420 | 0.5286 |
| homonyms_with_50_samples.json | braycurtis | 10 | 414/586 | 0.7065 |



## Метод bert_score, модель: sberbank-ai/sbert_large_nlu_ru

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 40/187 | 0.2139 |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 94/420 | 0.2238 |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 134/586 | 0.2287 |
| homonyms_ru_clean.json | euclidean | 26 | 114/187 | 0.6096 |
| homonyms_ru_dirty.json | euclidean | 57 | 207/420 | 0.4929 |
| homonyms_with_50_samples.json | euclidean | 10 | 387/586 | 0.6604 |
| homonyms_ru_clean.json | manhattan | 26 | 117/187 | 0.6257 |
| homonyms_ru_dirty.json | manhattan | 57 | 208/420 | 0.4952 |
| homonyms_with_50_samples.json | manhattan | 10 | 383/586 | 0.6536 |
| homonyms_ru_clean.json | minkowski | 26 | 114/187 | 0.6096 |
| homonyms_ru_dirty.json | minkowski | 57 | 207/420 | 0.4929 |
| homonyms_with_50_samples.json | minkowski | 10 | 387/586 | 0.6604 |
| homonyms_ru_clean.json | hamming | 26 | 88/187 | 0.4706 |
| homonyms_ru_dirty.json | hamming | 57 | 151/420 | 0.3595 |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 |
| homonyms_ru_clean.json | canberra | 26 | 112/187 | 0.5989 |
| homonyms_ru_dirty.json | canberra | 57 | 202/420 | 0.4810 |
| homonyms_with_50_samples.json | canberra | 10 | 369/586 | 0.6297 |
| homonyms_ru_clean.json | braycurtis | 26 | 116/187 | 0.6203 |
| homonyms_ru_dirty.json | braycurtis | 57 | 205/420 | 0.4881 |
| homonyms_with_50_samples.json | braycurtis | 10 | 378/586 | 0.6451 |


## Метод bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 37/187 | 0.1979 |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 79/420 | 0.1881 |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 133/586 | 0.2270 |
| homonyms_ru_clean.json | euclidean | 26 | 122/187 | 0.6524 |
| homonyms_ru_dirty.json | euclidean | 57 | 228/420 | 0.5429 |
| homonyms_with_50_samples.json | euclidean | 10 | 406/586 | 0.6928 |
| homonyms_ru_clean.json | manhattan | 26 | 126/187 | 0.6738 |
| homonyms_ru_dirty.json | manhattan | 57 | 228/420 | 0.5429 |
| homonyms_with_50_samples.json | manhattan | 10 | 406/586 | 0.6928 |
| homonyms_ru_clean.json | minkowski | 26 | 122/187 | 0.6524 |
| homonyms_ru_dirty.json | minkowski | 57 | 228/420 | 0.5429 |
| homonyms_with_50_samples.json | minkowski | 10 | 406/586 | 0.6928 |
| homonyms_ru_clean.json | hamming | 26 | 88/187 | 0.4706 |
| homonyms_ru_dirty.json | hamming | 57 | 151/420 | 0.3595 |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 |
| homonyms_ru_clean.json | canberra | 26 | 130/187 | 0.6952 |
| homonyms_ru_dirty.json | canberra | 57 | 232/420 | 0.5524 |
| homonyms_with_50_samples.json | canberra | 10 | 407/586 | 0.6945 |
| homonyms_ru_clean.json | braycurtis | 26 | 131/187 | 0.7005 |
| homonyms_ru_dirty.json | braycurtis | 57 | 241/420 | 0.5738 |
| homonyms_with_50_samples.json | braycurtis | 10 | 416/586 | 0.7099 |


## Метод bert_score, модель: DeepPavlov/rubert-base-cased-sentence

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 43/187 | 0.2299 |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 88/420 | 0.2095 |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 141/586 | 0.2406 |
| homonyms_ru_clean.json | euclidean | 26 | 118/187 | 0.6310 |
| homonyms_ru_dirty.json | euclidean | 57 | 227/420 | 0.5405 |
| homonyms_with_50_samples.json | euclidean | 10 | 389/586 | 0.6638 |
| homonyms_ru_clean.json | manhattan | 26 | 114/187 | 0.6096 |
| homonyms_ru_dirty.json | manhattan | 57 | 225/420 | 0.5357 |
| homonyms_with_50_samples.json | manhattan | 10 | 392/586 | 0.6689 |
| homonyms_ru_clean.json | minkowski | 26 | 118/187 | 0.6310 |
| homonyms_ru_dirty.json | minkowski | 57 | 227/420 | 0.5405 |
| homonyms_with_50_samples.json | minkowski | 10 | 389/586 | 0.6638 |
| homonyms_ru_clean.json | hamming | 26 | 88/187 | 0.4706 |
| homonyms_ru_dirty.json | hamming | 57 | 151/420 | 0.3595 |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 |
| homonyms_ru_clean.json | canberra | 26 | 116/187 | 0.6203 |
| homonyms_ru_dirty.json | canberra | 57 | 221/420 | 0.5262 |
| homonyms_with_50_samples.json | canberra | 10 | 390/586 | 0.6655 |
| homonyms_ru_clean.json | braycurtis | 26 | 119/187 | 0.6364 |
| homonyms_ru_dirty.json | braycurtis | 57 | 226/420 | 0.5381 |
| homonyms_with_50_samples.json | braycurtis | 10 | 391/586 | 0.6672 |




## Метод bert_score, модель: DeepPavlov/rubert-base-cased

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 59/187 | 0.3155 |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 144/420 | 0.3429 |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 242/586 | 0.4130 |
| homonyms_ru_clean.json | euclidean | 26 | 100/187 | 0.5348 |
| homonyms_ru_dirty.json | euclidean | 57 | 137/420 | 0.3262 |
| homonyms_with_50_samples.json | euclidean | 10 | 282/586 | 0.4812 |
| homonyms_ru_clean.json | manhattan | 26 | 97/187 | 0.5187 |
| homonyms_ru_dirty.json | manhattan | 57 | 135/420 | 0.3214 |
| homonyms_with_50_samples.json | manhattan | 10 | 287/586 | 0.4898 |
| homonyms_ru_clean.json | minkowski | 26 | 100/187 | 0.5348 |
| homonyms_ru_dirty.json | minkowski | 57 | 137/420 | 0.3262 |
| homonyms_with_50_samples.json | minkowski | 10 | 282/586 | 0.4812 |
| homonyms_ru_clean.json | hamming | 26 | 88/187 | 0.4706 |
| homonyms_ru_dirty.json | hamming | 57 | 151/420 | 0.3595 |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 |
| homonyms_ru_clean.json | canberra | 26 | 105/187 | 0.5615 |
| homonyms_ru_dirty.json | canberra | 57 | 162/420 | 0.3857 |
| homonyms_with_50_samples.json | canberra | 10 | 301/586 | 0.5137 |
| homonyms_ru_clean.json | braycurtis | 26 | 107/187 | 0.5722 |
| homonyms_ru_dirty.json | braycurtis | 57 | 148/420 | 0.3524 |
| homonyms_with_50_samples.json | braycurtis | 10 | 292/586 | 0.4983 |


## Метод bert_score, модель: inkoziev/sbert_synonymy

| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных |
| --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | similarity_cosine | 26 | 45/187 | 0.2406 |
| homonyms_ru_dirty.json | similarity_cosine | 57 | 93/420 | 0.2214 |
| homonyms_with_50_samples.json | similarity_cosine | 10 | 180/586 | 0.3072 |
| homonyms_ru_clean.json | euclidean | 26 | 106/187 | 0.5668 |
| homonyms_ru_dirty.json | euclidean | 57 | 174/420 | 0.4143 |
| homonyms_with_50_samples.json | euclidean | 10 | 327/586 | 0.5580 |
| homonyms_ru_clean.json | manhattan | 26 | 109/187 | 0.5829 |
| homonyms_ru_dirty.json | manhattan | 57 | 175/420 | 0.4167 |
| homonyms_with_50_samples.json | manhattan | 10 | 332/586 | 0.5666 |
| homonyms_ru_clean.json | minkowski | 26 | 106/187 | 0.5668 |
| homonyms_ru_dirty.json | minkowski | 57 | 174/420 | 0.4143 |
| homonyms_with_50_samples.json | minkowski | 10 | 327/586 | 0.5580 |
| homonyms_ru_clean.json | hamming | 26 | 88/187 | 0.4706 |
| homonyms_ru_dirty.json | hamming | 57 | 151/420 | 0.3595 |
| homonyms_with_50_samples.json | hamming | 10 | 261/586 | 0.4454 |
| homonyms_ru_clean.json | canberra | 26 | 113/187 | 0.6043 |
| homonyms_ru_dirty.json | canberra | 57 | 171/420 | 0.4071 |
| homonyms_with_50_samples.json | canberra | 10 | 333/586 | 0.5683 |
| homonyms_ru_clean.json | braycurtis | 26 | 109/187 | 0.5829 |
| homonyms_ru_dirty.json | braycurtis | 57 | 167/420 | 0.3976 |
| homonyms_with_50_samples.json | braycurtis | 10 | 329/586 | 0.5614 |

