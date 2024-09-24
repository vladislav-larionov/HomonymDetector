homonyms_with_50_samples.json

## Model: Word2Vec + MeanEmbeddingVectorizer + StandardScaler
RandomForest n_estimators=500 criterion=entropy bootstrap=False max_features=log2 | F1_micro_avg = 0.5696 | F1_macro_avg = 0.5531
RandomForest n_estimators=200 entropy bootstrap=False | F1_micro_avg = 0.5688 | F1_macro_avg = 0.5549

## Model: Word2Vec + MeanEmbeddingVectorizer
RandomForest entropy max_features=log2 bootstrap=False | F1_micro_avg = 0.5721 | F1_macro_avg = 0.5577

## TfidfVectorizer
SVM class_weight=balanced linear | F1_micro_avg = 0.7039 | F1_macro_avg = 0.6999

## BertTransformerEmbedding + cointegrated/rubert-tiny2
SVM kernel=poly coef0=0.75 | F1_micro_avg = 0.8401 | F1_macro_avg = 0.8359

## BertTransformerEmbedding + cointegrated/rubert-tiny

SVM class_weight=balanced kernel=poly coef0=0.75 | F1_micro_avg = 0.7743 | F1_macro_avg = 0.7686


## BertTransformerEmbedding + sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

SVM kernel=poly degree=4 coef0=0.75 | F1_micro_avg = 0.8509 | F1_macro_avg = 0.8464
SVM class_weight=balanced kernel=poly coef0=0.75 | F1_micro_avg = 0.8508 | F1_macro_avg = 0.8468

