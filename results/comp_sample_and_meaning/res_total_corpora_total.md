# /home/vladislav/projects/python/HomonymDetector/comp_sample_and_meaning/compare_sample_and_meaning.py

Корпус corpora.json

## Итог

| Метод | Метрика | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- |
| w2v_emb | braycurtis | 1354/2933 | 0.4616 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| navec_score | euclidean | 1801/2933 | 0.6140 | лемматизация = True, Удаление стоп-слов = True |
| gensim_pretrainde, модель = word2vec-ruscorpora-300 | canberra | 1792/2933 | 0.6110 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |
| d2v_emb | manhattan | 1402/2933 | 0.4780 | лемматизация = True, Удаление стоп-слов = True |
| bert_score, модель: cointegrated/rubert-tiny | canberra | 1659/2933 | 0.5656 |
| bert_score, модель: cointegrated/rubert-tiny2 | euclidean | 2004/2933 | 0.6833 |
| bert_score, модель: cointegrated/rubert-tiny2 | manhattan | 2005/2933 | 0.6836 |
| bert_score, модель: sberbank-ai/sbert_large_nlu_ru | euclidean | 1858/2933 | 0.6335 |
| bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | euclidean | 1938/2933 | 0.6608 |
| bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | manhattan | 1957/2933 | 0.6672 |
| bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | canberra | 1980/2933 | 0.6751 |
| bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | braycurtis | 1992/2933 | 0.6792 |
| bert_score, модель: DeepPavlov/rubert-base-cased-sentence | euclidean | 1894/2933 | 0.6458 |
| bert_score, модель: DeepPavlov/rubert-base-cased-sentence | manhattan | 1895/2933 | 0.6461 |
| bert_score, модель: DeepPavlov/rubert-base-cased | canberra | 1554/2933 | 0.5298 |
| bert_score, модель: inkoziev/sbert_synonymy | euclidean | 1653/2933 | 0.5636 |
| bert_score, модель: inkoziev/sbert_synonymy | manhattan | 1644/2933 | 0.5605 |


## Метод w2v_emb

| Метод | Метрика | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- |
| w2v_emb | braycurtis | 1354/2933 | 0.4616 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |

## Метод navec_score

| Метод | Метрика | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- |
| navec_score | euclidean | 1801/2933 | 0.6140 | лемматизация = True, Удаление стоп-слов = True |

## Метод gensim_pretrainde, модель = word2vec-ruscorpora-300

| Метод | Метрика | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- |
| gensim_pretrainde, модель = word2vec-ruscorpora-300 | canberra | 1792/2933 | 0.6110 | лемматизация = True, Удаление стоп-слов = False, Вектор - среднеарифметическое значение поэлементно |

## Метод d2v_emb

| Метод | Метрика | Соотношение | Доля угаданных | Параметры |
| --- | --- | --- | --- | --- |
| d2v_emb | manhattan | 1402/2933 | 0.4780 | лемматизация = True, Удаление стоп-слов = True |

## Метод bert_score, модель: cointegrated/rubert-tiny

| Метод | Метрика | Соотношение | Доля угаданных |
| --- | --- | --- | --- |
| bert_score, модель: cointegrated/rubert-tiny | canberra | 1659/2933 | 0.5656 |

## Метод bert_score, модель: cointegrated/rubert-tiny2

| Метод | Метрика | Соотношение | Доля угаданных |
| --- | --- | --- | --- |
| bert_score, модель: cointegrated/rubert-tiny2 | euclidean | 2004/2933 | 0.6833 |
| bert_score, модель: cointegrated/rubert-tiny2 | manhattan | 2005/2933 | 0.6836 |

## Метод bert_score, модель: sberbank-ai/sbert_large_nlu_ru

| Метод | Метрика | Соотношение | Доля угаданных |
| --- | --- | --- | --- |
| bert_score, модель: sberbank-ai/sbert_large_nlu_ru | euclidean | 1858/2933 | 0.6335 |

## Метод bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

| Метод | Метрика | Соотношение | Доля угаданных |
| --- | --- | --- | --- |
| bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | euclidean | 1938/2933 | 0.6608 |
| bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | manhattan | 1957/2933 | 0.6672 |
| bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | canberra | 1980/2933 | 0.6751 |
| bert_score, модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | braycurtis | 1992/2933 | 0.6792 |

## Метод bert_score, модель: DeepPavlov/rubert-base-cased-sentence

| Метод | Метрика | Соотношение | Доля угаданных |
| --- | --- | --- | --- |
| bert_score, модель: DeepPavlov/rubert-base-cased-sentence | euclidean | 1894/2933 | 0.6458 |
| bert_score, модель: DeepPavlov/rubert-base-cased-sentence | manhattan | 1895/2933 | 0.6461 |

## Метод bert_score, модель: DeepPavlov/rubert-base-cased

| Метод | Метрика | Соотношение | Доля угаданных |
| --- | --- | --- | --- |
| bert_score, модель: DeepPavlov/rubert-base-cased | canberra | 1554/2933 | 0.5298 |

## Метод bert_score, модель: inkoziev/sbert_synonymy

| Метод | Метрика | Соотношение | Доля угаданных |
| --- | --- | --- | --- |
| bert_score, модель: inkoziev/sbert_synonymy | euclidean | 1653/2933 | 0.5636 |
| bert_score, модель: inkoziev/sbert_synonymy | manhattan | 1644/2933 | 0.5605 |






