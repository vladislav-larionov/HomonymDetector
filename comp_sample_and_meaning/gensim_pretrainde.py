import sys

import gensim.downloader as api
# https://www.kaggle.com/code/ksenialisitsina/text-vectors
import json
import string
from pprint import pprint

import numpy as np
import pymorphy3

import snowballstemmer
from gensim.models import Word2Vec, Phrases
from gensim.models.keyedvectors import load_word2vec_format, KeyedVectors
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

from io_utils import read_and_filter_words
from similarity_metrics.cosine import similarity_cosine_w2v, similarity_cosine_numpy
from similarity_metrics.distance_metric import compare_by_sklearn


morph = pymorphy3.MorphAnalyzer()

def text_to_words(text, use_lemma=True, remove_stop_words=True):
    tokens = word_tokenize(text)
    stop_words = set(string.punctuation)
    if remove_stop_words:
        stop_words.update(set(stopwords.words("russian")))
    if use_lemma:
        return [morph.parse(word)[0].normal_form  + f"_{morph.parse(word)[0].tag.POS}" for word in tokens if word not in stop_words]
    else:
        return [word + f"_{morph.parse(word)[0].tag.POS}" for word in tokens if word not in stop_words]


def sum_vectors(words, model):
    sample_sum_vec = None
    for word in words:
        if word in model:
            if sample_sum_vec is not None:
                sample_sum_vec = np.add(sample_sum_vec, model.get_vector(word))
            else:
                sample_sum_vec = model.get_vector(word)
    return sample_sum_vec


def words_to_vectors(model, words, mean=False):
    sum_samples = []
    for i, sample in enumerate(words):
        try:
            if mean:
                vector = model.get_mean_vector(sample, ignore_missing=True)
            else:
                vector = sum_vectors(sample, model)
            if vector is not None:
                sum_samples.append((i, vector))
        except ValueError as err:
            print("words_to_vectors " + "" + str(sample), file=sys.stderr)
            print(err)
    return sum_samples

def compare_with_cosine_similarity(model, valid_words, ambiguity_filtered_by_3_samples, use_lemma=True,
                                   remove_stop_words=True, log=False, metric="euclidean", vect_act_mean=False):
    total = 0
    total_word = 0
    total_used_word = 0
    right = 0
    for mord_num, word in enumerate(valid_words):
        total_word += 1
        word_data = ambiguity_filtered_by_3_samples[word]
        samples = [text_to_words(sample['text'], use_lemma, remove_stop_words) for sample in word_data['samples']]
        meanings = [text_to_words(meaning['определение'], use_lemma, remove_stop_words) for meaning in word_data['meanings']]
        sum_samples = words_to_vectors(model, samples, vect_act_mean)
        sum_meanings = words_to_vectors(model, meanings, vect_act_mean)
        used = False
        for i, sample in enumerate(sum_samples):
            if sum_meanings:
                total += 1
                used = True
                if log:
                    print("Слово: ", word)
                    print("Пример: ", word_data['samples'][sample[0]]['text'])
                if metric == 'similarity_cosine':
                    meaning = list(sorted(sum_meanings, key=lambda _meaning: similarity_cosine_numpy(sample[1], _meaning[1])))[0]
                else:
                    meaning = list(sorted(sum_meanings, key=lambda _meaning: compare_by_sklearn(sample[1], _meaning[1], metric=metric)))[0]
                if log:
                    print("Значение: ", word_data['meanings'][meaning[0]]['определение'])
                    print("Верное значение: ", word_data['meanings'][word_data['samples'][sample[0]]['meaning']]['определение'])
                    print("Верно: ", word_data['samples'][sample[0]]["meaning"] == meaning[0])
                if word_data['samples'][sample[0]]["meaning"] == meaning[0]:
                    right += 1
            if log:
                print("___")
        if used:
            total_used_word += 1
        if log:
            print("__________________________________")
    return dict(right=right, total=total, total_word=total_word)


def gensim_pretrainde(filename, file=sys.stdout):
    print(f"## Метод gensim_pretrainde, модель = word2vec-ruscorpora-300\n", file=file)
    print(f"| Корпус | Метрика | Всего слов | Соотношение | Доля угаданных | Параметры |", file=file)
    print(f"| --- | --- | --- | --- | --- | --- |", file=file)
    with open(f"../dicts/{filename}") as ambiguity_filtered_by_3_samples_json:
        ambiguity_filtered_by_3_samples = json.load(ambiguity_filtered_by_3_samples_json)
        valid_words = read_and_filter_words(ambiguity_filtered_by_3_samples)
        model = api.load("word2vec-ruscorpora-300")
        for metric in ['similarity_cosine', "euclidean", "manhattan", "minkowski", "hamming", "canberra", "braycurtis"]:
            param_list = [
                dict(use_lemma=False, remove_stop_words=False, vect_act_mean=True),
                dict(use_lemma=False, remove_stop_words=False, vect_act_mean=False),
                dict(use_lemma=True, remove_stop_words=False, vect_act_mean=True),
                dict(use_lemma=True, remove_stop_words=False, vect_act_mean=False),
                dict(use_lemma=False, remove_stop_words=True, vect_act_mean=True),
                dict(use_lemma=False, remove_stop_words=True, vect_act_mean=False),
                dict(use_lemma=True, remove_stop_words=True, vect_act_mean=True),
                dict(use_lemma=True, remove_stop_words=True, vect_act_mean=False),
            ]
            for params in param_list:
                vect_act_mean = params['vect_act_mean']
                if vect_act_mean:
                    vect = "Вектор - среднеарифметическое значение поэлементно"
                else:
                    vect = "Вектор - сумма значений поэлементно"
                statistic = compare_with_cosine_similarity(model, valid_words, ambiguity_filtered_by_3_samples, metric=metric, **params)
                print(f"| {filename} | {metric} | {statistic['total_word']} | {statistic['right']}/{statistic['total']} | {statistic['right'] / statistic['total']:.4f} | лемматизация = {params['use_lemma']}, Удаление стоп-слов = {params['remove_stop_words']}, {vect} |", file=file)

    print("________________________________________")


def main():
    # filename = "homonyms_with_50_samples.json"
    # filename = "narusco_ru.json"
    filename = "homonyms_ru_clean.json"
    gensim_pretrainde(filename)


if __name__ == "__main__":
    main()
