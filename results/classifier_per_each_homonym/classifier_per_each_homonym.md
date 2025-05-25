# Отдельный классификатор для каждого омонима

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
Примеры переводились в вектора, каждому из которых ставилась в соответствие метка - номер определения.
Далее 60% примеров бралось на обучение и 40% на тест.
Для классификации использовались классификаторы svm, RandomForest, K Neighbors с различными параметрами.

Для преобразования текстов в вектора использовались различные эммбединги: w2v, tfidf, bert различных моделей: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, cointegrated/rubert-tiny2, cointegrated/rubert-tiny.

Корпуса:
* homonyms_ru_clean.json - корпус вначале собранный с сайта homonyms.ru и потом вручную очищенн.
* homonyms_with_50_samples.json - корпус собранный с различных сайтов вручную. Каждый омоним имеет минимум по 20 примеров и не менее 50 примеров в общем.

С корпусом homonyms_ru_dirty.json классификаторы не завелись из-за нехватки данных для разметки.

## Выводы

Здесь также, как и в предыдущем разделе лучше всего справились bert-модели paraphrase-multilingual-MiniLM-L12-v2 и rubert-tiny2 с результатом 0.8359 - 0.8509 средних микро- и макро-f-меры на корпусе homonyms_with_50_samples. У второго корпуса результаты отстают на 30-40%.
Это можно объяснить тем, что корпус homonyms_with_50_samples имеет больше примеров на каждый омоним, что позитивно влияет на обучение. Кроме того, корпус homonyms_ru_clean имеет больше разлиных омонимов с меньшим количеством примеров, что сильно сказывается на показателях метрик.


## Метод w2v

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_ru_clean.json | RandomForest | 0.3704 |0.4392 | n_estimators=150 entropy bootstrap=False |
| homonyms_with_50_samples.json | RandomForest |  0.5608 | 0.5762 | n_estimators=150 entropy bootstrap=False |


## Метод Tfidf

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | SVM | 0.6999 | 0.7039 | class_weight=balanced linear |
| homonyms_ru_clean.json | RandomForest | 0.4332  | 0.3237 | entropy max_features=log2 |


## Метод bert_score, модель: rubert-tiny

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | SVM | 0.7686 | 0.7743 | class_weight=balanced kernel=poly coef0=0.75 |
| homonyms_ru_clean.json | SVM | 0.4330 | 0.5165 | class_weight=balanced linear | 


## Метод bert_score, модель: paraphrase-multilingual-MiniLM-L12-v2

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | SVM | 0.8464 | 0.8509 | kernel=poly degree=4 coef0=0.75  |
| homonyms_ru_clean.json | RandomForest | 0.3171 | 0.4219 | n_estimators=150 entropy bootstrap=False |

## Метод bert_score, модель: rubert-tiny2

| Корпус | Классификатор | F1_macro_avg | F1_micro_avg | Параметры |
| --- | --- | --- | --- | --- | --- |
| homonyms_with_50_samples.json | SVM | 0.8359 | 0.8401 | class_weight=balanced kernel=poly coef0=0.75 |
| homonyms_ru_clean.json | SVM |  0.4158 | 0.5088 | class_weight=balanced linear |
