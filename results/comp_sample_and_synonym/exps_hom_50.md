## w2v_emb

## navec_score

## gensim_pretrainde

## d2v_emb

metric = euclidean
use_lemma = False
remove_stop_words = True
Total: 265/586 0.4522
Total used words: 10/10

## bertscore cointegrated/rubert-tiny

model cointegrated/rubert-tiny
metric manhattan
Total: 312/586 0.5324
Total used words: 10/10


## bertscore cointegrated/rubert-tiny2

model cointegrated/rubert-tiny2
metric braycurtis
Total: 384/586 0.6553
Total used words: 10/10


## bertscore sberbank-ai/sbert_large_nlu_ru

model sberbank-ai/sbert_large_nlu_ru
metric euclidean
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 368/586 0.6280
Total used words: 10/10


## bertscore sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
metric euclidean
Total: 374/586 0.6382
Total used words: 10/10

model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
metric braycurtis
Total: 377/586 0.6433
Total used words: 10/10

## bertscore DeepPavlov/rubert-base-cased-sentence

model DeepPavlov/rubert-base-cased-sentence
metric manhattan
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 385/586 0.6570
Total used words: 10/10

model DeepPavlov/rubert-base-cased-sentence
metric canberra
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 386/586 0.6587
Total used words: 10/10

model DeepPavlov/rubert-base-cased-sentence
metric braycurtis
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 386/586 0.6587
Total used words: 10/10

## bertscore DeepPavlov/rubert-base-cased

model DeepPavlov/rubert-base-cased
metric manhattan
Some weights of the model checkpoint at DeepPavlov/rubert-base-cased were not used when initializing BertModel: ['cls.predictions.bias', 'cls.predictions.decoder.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Total: 328/586 0.5597
Total used words: 10/10

## bertscore inkoziev/sbert_synonymy

model inkoziev/sbert_synonymy
metric manhattan
Total: 319/586 0.5444
Total used words: 10/10

