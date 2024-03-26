import json
import string
from pprint import pprint

import snowballstemmer
from gensim.models import Word2Vec, Phrases
from gensim.models.keyedvectors import load_word2vec_format, KeyedVectors
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

from io_utils import read_and_filter_words
from similarity_metrics.cosine import similarity_cosine_w2v, similarity_cosine_numpy


def get_word_texts_as_sentences(ambiguity_filtered_by_3_samples, words) -> list:
    sentences = []
    stop_words = set(stopwords.words("russian"))
    stop_words.update(set(string.punctuation))
    for word in words:
        for sample in ambiguity_filtered_by_3_samples[word]["samples"]:
            if sample["адекватность"] and sample["meaning"] is not None:
                for sent in sent_tokenize(sample["text"]):
                    tokens = word_tokenize(sent)
                    filtered_tokens = [word for word in tokens if word not in stop_words]
                    sentences.append(filtered_tokens)
    return sentences


def text_to_words(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("russian"))
    stop_words.update(set(string.punctuation))
    return [word for word in tokens if word not in stop_words]


def sum_vectors(words, model):
    sample_sum_vec = None
    for word in words:
        if word in model:
            if sample_sum_vec is not None:
                sample_sum_vec = sample_sum_vec + model.get_vector(word)
            else:
                sample_sum_vec = model.get_vector(word)
    return sample_sum_vec


def words_to_vectors(model, words):
    sum_samples = []
    for i, sample in enumerate(words):
        try:
            vector = model.get_mean_vector(sample[1:], ignore_missing=True)
            # vector = sum_vectors(sample[1:], model)
            if vector is not None:
                sum_samples.append((i, vector))
        except ValueError as err:
            print(sample)
    return sum_samples


def compare_with_cosine_similarity(model, valid_words, ambiguity_filtered_by_3_samples, log=False):
    # similarity_metrics(model.wv['space'], model.wv['france'])
    total = 0
    total_word = 0
    total_used_word = 0
    right = 0
    for mord_num, word in enumerate(valid_words):
        total_word += 1
        word_data = ambiguity_filtered_by_3_samples[word]
        samples = [text_to_words(sample['text']) for sample in word_data['samples']]
        meanings = [text_to_words(meaning['определение']) for meaning in word_data['meanings']]
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


# sum_vectors
#    ambiguity_filtered_by_3_samples
#       Total: 202/478
#       Total used words: 25/32
#    homonyms_ru
#       Total: 362/810
#       Total used words: 106/116
# get_mean_vector
#    ambiguity_filtered_by_3_samples
#       Total: 125/522
#       Total used words: 32/32
#    homonyms_ru
#       Total: 399/856
#       Total used words: 116/116
def main():
    filename = "ambiguity_filtered_by_3_samples.json"
    # filename = "homonyms_ru.json"
    print(filename)
    with open(filename) as ambiguity_filtered_by_3_samples_json:
        ambiguity_filtered_by_3_samples = json.load(ambiguity_filtered_by_3_samples_json)
        valid_words = read_and_filter_words(ambiguity_filtered_by_3_samples)
        print(f"Valid number {len(valid_words)}")
        # for word in valid_words:
        #     print(word)
        sentences = get_word_texts_as_sentences(ambiguity_filtered_by_3_samples, valid_words)

        # bigram_transformer = Phrases(sentences)
        # model = Word2Vec(sentences=bigram_transformer[sentences], vector_size=100, window=5, min_count=1, workers=4, sg=1).wv
        model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1).wv
        # model = load_word2vec_format("model.hdf5")
        # print(model.get_vector('жалоба'))
        # model.train()
        # model.save("word2vec.model")
        stemmer = snowballstemmer.stemmer('russian')
        # print(model.wv[stemmer.stemWords(['дешевый'])[0]])
        compare_with_cosine_similarity(model, valid_words, ambiguity_filtered_by_3_samples)
        # print(model.wv['жалоба'])
        # pprint(model.wv.most_similar('жалоба', topn=10))


if __name__ == "__main__":
    main()
