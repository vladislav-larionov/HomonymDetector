## w2v_emb

metric = manhattan
use_lemma = True
remove_stop_words = False
get_mean_vector
Total: 103/187 0.5508
Total used words: 26

## navec_score

use_lemma = False
remove_stop_words = True
Total: 35/187 0.1872
Total used words: 26/26

## gensim_pretrainde

Model: word2vec-ruscorpora-300

metric = canberra
use_lemma = True
remove_stop_words = False
sum_vectors
Total: 113/187 0.6043

## d2v_emb

metric = braycurtis
use_lemma = False
remove_stop_words = True
Total: 96/187 0.5134
Total used words: 26/26


## bertscore cointegrated/rubert-tiny

model cointegrated/rubert-tiny
metric canberra
Total: 107/187 0.5722
Total used words: 26/26


## bertscore cointegrated/rubert-tiny2

model cointegrated/rubert-tiny2
metric euclidean
Total: 128/187 0.6845
Total used words: 26/26

model cointegrated/rubert-tiny2
metric manhattan
Total: 129/187 0.6898
Total used words: 26/26


## bertscore sberbank-ai/sbert_large_nlu_ru

model sberbank-ai/sbert_large_nlu_ru
metric manhattan
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 117/187 0.6257
Total used words: 26/26


## bertscore sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
metric manhattan
Total: 126/187 0.6738
Total used words: 26/26

model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
metric canberra
Total: 130/187 0.6952
Total used words: 26/26

model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
metric braycurtis
Total: 131/187 0.7005
Total used words: 26/26


## bertscore DeepPavlov/rubert-base-cased-sentence

model DeepPavlov/rubert-base-cased-sentence
metric euclidean
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 118/187 0.6310
Total used words: 26/26


model DeepPavlov/rubert-base-cased-sentence
metric braycurtis
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 119/187 0.6364
Total used words: 26/26

## bertscore DeepPavlov/rubert-base-cased

model DeepPavlov/rubert-base-cased
metric euclidean
Some weights of the model checkpoint at DeepPavlov/rubert-base-cased were not used when initializing BertModel: ['cls.predictions.bias', 'cls.predictions.decoder.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 100/187 0.5348
Total used words: 26/26

model DeepPavlov/rubert-base-cased
metric canberra
Some weights of the model checkpoint at DeepPavlov/rubert-base-cased were not used when initializing BertModel: ['cls.predictions.bias', 'cls.predictions.decoder.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 105/187 0.5615
Total used words: 26/26

model DeepPavlov/rubert-base-cased
metric braycurtis
Some weights of the model checkpoint at DeepPavlov/rubert-base-cased were not used when initializing BertModel: ['cls.predictions.bias', 'cls.predictions.decoder.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 107/187 0.5722
Total used words: 26/26

## bertscore inkoziev/sbert_synonymy

model inkoziev/sbert_synonymy
metric canberra
Total: 113/187 0.6043
Total used words: 26/26
