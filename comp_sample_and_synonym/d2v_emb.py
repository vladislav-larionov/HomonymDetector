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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

morph = pymorphy3.MorphAnalyzer()


def get_word_texts_as_sentences(ambiguity_filtered_by_3_samples, words, use_lemma=True, remove_stop_words=True) -> list:
    sentences = []
    stop_words = set(stopwords.words("russian"))
    stop_words.update(set(string.punctuation))
    for word in words:
        for sample in ambiguity_filtered_by_3_samples[word]["samples"]:
            for sent in sent_tokenize(sample["text"]):
                tokens = word_tokenize(sent)
                if remove_stop_words:
                    tokens = [word for word in tokens if word not in stop_words]
                if use_lemma:
                    tokens = [morph.parse(word)[0].normal_form for word in tokens]
                sentences.append(tokens)
    return sentences


def text_to_words(text, use_lemma=True, remove_stop_words=True):
    tokens = word_tokenize(text)
    stop_words = set(string.punctuation)
    if remove_stop_words:
        stop_words.update(set(stopwords.words("russian")))
    if use_lemma:
        return [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]
    else:
        return [word for word in tokens if word not in stop_words]


def sum_vectors(words, model):
    sample_sum_vec = None
    for word in words:
        if word in model:
            if sample_sum_vec is not None:
                sample_sum_vec = np.add(sample_sum_vec, model.get_vector(word))
            else:
                sample_sum_vec = model.get_vector(word)
    return sample_sum_vec


def words_to_vectors(model: Doc2Vec, words):
    sum_samples = []
    for i, sample in enumerate(words):
        try:
            vector = model.infer_vector(sample)
            if vector is not None:
                sum_samples.append((i, vector))
        except ValueError as err:
            print("words_to_vectors " + "" + str(sample))
            print(err)
    return sum_samples


def compare_with_cosine_similarity(model, valid_words, ambiguity_filtered_by_3_samples, use_lemma=True,
                                   remove_stop_words=True, log=False, metric="euclidean"):
    total = 0
    total_word = 0
    total_used_word = 0
    right = 0
    for mord_num, word in enumerate(valid_words):
        total_word += 1
        word_data = ambiguity_filtered_by_3_samples[word]
        samples = [text_to_words(sample['text'], use_lemma, remove_stop_words) for sample in word_data['samples']]
        meanings = [text_to_words(meaning['synonym']['определение'], use_lemma, remove_stop_words) for meaning in word_data['meanings']]
        sum_samples = words_to_vectors(model, samples)
        sum_meanings = words_to_vectors(model, meanings)
        used = False
        for i, sample in enumerate(sum_samples):
            if sum_meanings:
                total += 1
                used = True
                if log:
                    print("Слово: ", word)
                    print("Пример: ", word_data['samples'][sample[0]]['text'])
                meaning = list(sorted(sum_meanings, key=lambda _meaning: compare_by_sklearn(sample[1], _meaning[1], metric=metric)))[0]
                if log:
                    print("Значение: ", word_data['meanings'][meaning[0]]['определение'])
                    print("Верное значение: ", word_data['meanings'][word_data['samples'][sample[0]]['meaning']]['определение'])
                    print("Синоним: ", word_data['meanings'][word_data['samples'][sample[0]]['meaning']]['synonym']['word'])
                    print("Синоним определение: ", word_data['meanings'][word_data['samples'][sample[0]]['meaning']]['synonym']['определение'])
                    print("Верно: ", word_data['samples'][sample[0]]["meaning"] == meaning[0])
                if word_data['samples'][sample[0]]["meaning"] == meaning[0]:
                    right += 1
            if log:
                print("___")
        if used:
            total_used_word += 1
        if log:
            print("__________________________________")
    print(f"Total: {right}/{total} {right/total:.4f}")
    print(f"Total used words: {total_used_word}/{total_word}")


def d2v_emb(filename):
    print("d2v_emb")
    print(filename)
    with open(f"../dicts/{filename}") as ambiguity_filtered_by_3_samples_json:
        ambiguity_filtered_by_3_samples = json.load(ambiguity_filtered_by_3_samples_json)
        valid_words = read_and_filter_words(ambiguity_filtered_by_3_samples)
        for metric in ["euclidean",
                       "manhattan", "minkowski", "hamming", "canberra", "braycurtis"
                       ]:
            param_list = [
                dict(use_lemma=False, remove_stop_words=False),
                dict(use_lemma=True, remove_stop_words=False),
                dict(use_lemma=False, remove_stop_words=True),
                dict(use_lemma=True, remove_stop_words=True),
            ]
            for params in param_list:
                sentences = get_word_texts_as_sentences(ambiguity_filtered_by_3_samples, valid_words, **params)
                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
                model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

                print(f"metric = {metric}")
                print(f"use_lemma = {params['use_lemma']}")
                print(f"remove_stop_words = {params['remove_stop_words']}")
                compare_with_cosine_similarity(model, valid_words, ambiguity_filtered_by_3_samples, metric=metric, **params)
                print()
    print("________________________________________")


def main():
    filename = "homonyms_with_50_samples.json"
    # filename = "narusco_ru.json"
    # filename = "homonyms_ru.json"
    d2v_emb(filename)


if __name__ == "__main__":
    main()
