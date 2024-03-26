import json

import torch
from transformers import AutoTokenizer, AutoModel

from io_utils import read_and_filter_words
from similarity_metrics.cosine import similarity_cosine_w2v, similarity_cosine_numpy


def create_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.cuda()  # uncomment it if you have a GPU
    return model, tokenizer


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def texts_to_vectors(texts, model, tokenizer):
    sum_samples = []
    for i, text in enumerate(texts):
        try:
            vector = embed_bert_cls(text, model, tokenizer)
            if vector is not None:
                sum_samples.append((i, vector))
        except ValueError as err:
            print(text)
    return sum_samples


def compare_with_cosine_similarity(valid_words, ambiguity_filtered_by_3_samples, model_name, log=False):
    model, tokenizer = create_model_and_tokenizer(model_name)
    total = 0
    total_word = 0
    total_used_word = 0
    right = 0
    for mord_num, word in enumerate(valid_words):
        total_word += 1
        word_data = ambiguity_filtered_by_3_samples[word]
        samples = [sample['text'] for sample in word_data['samples']]
        meanings = [meaning['определение'] for meaning in word_data['meanings']]
        sum_samples = texts_to_vectors(samples, model, tokenizer)
        sum_meanings = texts_to_vectors(meanings, model, tokenizer)
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


# cointegrated/rubert-tiny
#    ambiguity_filtered_by_3_samples
#       Total: 139/522
#       Total used words: 32/32
#    homonyms_ru
#       Total: 309/859
#       Total used words: 116/116
# cointegrated/rubert-tiny2
#    ambiguity_filtered_by_3_samples
#       Total: 83/522
#       Total used words: 32/32
#    homonyms_ru
#       Total: 239/859
#       Total used words: 116/116
# sberbank-ai/sbert_large_nlu_ru
#    ambiguity_filtered_by_3_samples
#       Total: 115/522
#       Total used words: 32/32
#    homonyms_ru
#       Total: 238/859
#       Total used words: 116/116
# DeepPavlov/rubert-base-cased-sentence
#    ambiguity_filtered_by_3_samples
#       Total: 119/522
#       Total used words: 32/32
#    homonyms_ru
#       Total: 226/859
#       Total used words: 116/116
# DeepPavlov/rubert-base-cased  Some weights of the model checkpoint
#    ambiguity_filtered_by_3_samples
#       Total: 113/522
#       Total used words: 32/32
#    homonyms_ru
#       Total: 353/859
#       Total used words: 116/116
# inkoziev/sbert_synonymy
#    ambiguity_filtered_by_3_samples
#       Total: 187/522
#       Total used words: 32/32
#    homonyms_ru
#       Total: 293/859
#       Total used words: 116/116
def bert_score():
    filename = "ambiguity_filtered_by_3_samples.json"
    # filename = "homonyms_ru.json"
    print("bert_score")
    print(filename)
    with open(filename) as ambiguity_filtered_by_3_samples_json:
        ambiguity_filtered_by_3_samples = json.load(ambiguity_filtered_by_3_samples_json)
        valid_words = read_and_filter_words(ambiguity_filtered_by_3_samples)
        print(f"Valid number {len(valid_words)}")
        compare_with_cosine_similarity(valid_words, ambiguity_filtered_by_3_samples, "cointegrated/rubert-tiny2")


if __name__ == "__main__":
    bert_score()
