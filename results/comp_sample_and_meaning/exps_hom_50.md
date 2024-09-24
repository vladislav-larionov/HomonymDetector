## w2v_emb
metric = canberra
use_lemma = False
remove_stop_words = False
get_mean_vector
Total: 241/525 0.4590
Total used words: 10/10

## navec_score
use_lemma = True
remove_stop_words = False
Total: 141/525 0.2686
Total used words: 10/10

## gensim_pretrainde
metric = braycurtis
use_lemma = True
remove_stop_words = True
sum_vectors
Total: 339/525 0.6457
Total used words: 10/10

## d2v_emb
metric = manhattan
use_lemma = True
remove_stop_words = True
Total: 246/525 0.4686
Total used words: 10/10


## bertscore cointegrated/rubert-tiny

model cointegrated/rubert-tiny
metric canberra
Total: 304/525 0.5790
Total used words: 10/10

model cointegrated/rubert-tiny2
metric euclidean
Total: 353/525 0.6724
Total used words: 10/10

## bertscore cointegrated/rubert-tiny2
model cointegrated/rubert-tiny2
metric manhattan
Total: 360/525 0.6857
Total used words: 10/10

model cointegrated/rubert-tiny2
metric canberra
Total: 366/525 0.6971
Total used words: 10/10

model cointegrated/rubert-tiny2
metric braycurtis
Total: 369/525 0.7029
Total used words: 10/10

## bertscore sberbank-ai/sbert_large_nlu_ru
model sberbank-ai/sbert_large_nlu_ru
metric euclidean
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 351/525 0.6686
Total used words: 10/10


## bertscore DeepPavlov/rubert-base-cased-sentence

model DeepPavlov/rubert-base-cased-sentence
metric canberra
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 349/525 0.6648
Total used words: 10/10

model DeepPavlov/rubert-base-cased-sentence
metric braycurtis
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 349/525 0.6648
Total used words: 10/10

## bertscore DeepPavlov/rubert-base-cased

model DeepPavlov/rubert-base-cased
metric canberra
Some weights of the model checkpoint at DeepPavlov/rubert-base-cased were not used when initializing BertModel: ['cls.predictions.bias', 'cls.predictions.decoder.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 275/525 0.5238
Total used words: 10/10


## bertscore inkoziev/sbert_synonymy

model inkoziev/sbert_synonymy
metric manhattan
Total: 296/525 0.5638
Total used words: 10/10

## bertscore sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

homonyms_with_50_samples.json
model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
metric euclidean
Total: 374/525 0.7124
Total used words: 10/10

model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
metric braycurtis
Total: 377/525 0.7181
Total used words: 10/10


