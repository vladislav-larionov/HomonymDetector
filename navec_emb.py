import json
import string

import pymorphy2
from navec import Navec
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from io_utils import read_and_filter_words
from similarity_metrics.cosine import similarity_cosine_numpy
from w2v_emb import morph


morph = pymorphy2.MorphAnalyzer()

def get_word_texts_as_sentences(ambiguity_filtered_by_3_samples, words, use_lemma=True, remove_stop_words=True) -> list:
    sentences = []
    stop_words = set(stopwords.words("russian"))
    stop_words.update(set(string.punctuation))
    for word in words:
        for sample in ambiguity_filtered_by_3_samples[word]["samples"]:
            if sample["адекватность"] and sample["meaning"] is not None:
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
                sample_sum_vec = sample_sum_vec + model.get(word)
            else:
                sample_sum_vec = model.get(word)
    return sample_sum_vec


def words_to_vectors(model, words):
    sum_samples = []
    for i, sample in enumerate(words):
        try:
            # vector = model.get_mean_vector(sample, ignore_missing=True)
            vector = sum_vectors(sample, model)
            if vector is not None:
                sum_samples.append((i, vector))
        except ValueError as err:
            print(sample)
    return sum_samples


def compare_with_cosine_similarity(model, valid_words, ambiguity_filtered_by_3_samples, use_lemma=True, remove_stop_words=True, log=False):
    # similarity_metrics(model.wv['space'], model.wv['france'])
    total = 0
    total_word = 0
    total_used_word = 0
    right = 0
    for mord_num, word in enumerate(valid_words):
        total_word += 1
        word_data = ambiguity_filtered_by_3_samples[word]
        samples = [text_to_words(sample['text'], use_lemma, remove_stop_words) for sample in word_data['samples']]
        meanings = [text_to_words(meaning['определение'], use_lemma, remove_stop_words) for meaning in word_data['meanings']]
        sum_samples = words_to_vectors(model, samples)
        sum_meanings = words_to_vectors(model, meanings)
        used = False
        for i, sample in enumerate(sum_samples):
            if sum_meanings and word_data['samples'][i]["адекватность"] and word_data['samples'][i]["meaning"] is not None:
                total += 1
                used = True
                if log:
                    print("Слово: ", word)
                    print("Пример: ", word_data['samples'][sample[0]]['text'])
                meaning = list(sorted(sum_meanings, key=lambda _meaning: similarity_cosine_numpy(sample[1], _meaning[1])))[0]
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
    print(f"Total: {right}/{total}")
    print(f"Total used words: {total_used_word}/{total_word}")



def navec_score():
    # filename = "ambiguity_filtered_by_3_samples.json"
    filename = "homonyms_ru.json"
    print("navec_score")
    print(filename)
    with open(filename) as ambiguity_filtered_by_3_samples_json:
        ambiguity_filtered_by_3_samples = json.load(ambiguity_filtered_by_3_samples_json)
        valid_words = read_and_filter_words(ambiguity_filtered_by_3_samples)
        path = 'models/navec_hudlit_v1_12B_500K_300d_100q.tar'
        navec = Navec.load(path)
        param_list = [
            dict(use_lemma=False, remove_stop_words=False),
            dict(use_lemma=True, remove_stop_words=False),
            dict(use_lemma=False, remove_stop_words=True),
            dict(use_lemma=True, remove_stop_words=True),
        ]
        for params in param_list:
            print(f"use_lemma = {params['use_lemma']}")
            print(f"remove_stop_words = {params['remove_stop_words']}")
            compare_with_cosine_similarity(navec, valid_words, ambiguity_filtered_by_3_samples, **params)
            print()


if __name__ == "__main__":
    navec_score()
