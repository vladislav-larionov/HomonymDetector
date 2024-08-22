import json
import warnings
from operator import itemgetter

from gensim.models import Word2Vec
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classifier_per_each_homonym.bert_transformer_embedding import BertTransformerEmbedding
from classifier_per_each_homonym.mean_embedding_vectorizer import MeanEmbeddingVectorizer



def print_statistics(trues, res, label=None):
    if label:
        print(f'Statistics of {label}:')
    print(f'A:\t\t\t{accuracy_score(trues, res):1.4f}')
    print(f'P_micro:\t{precision_score(trues, res, average="micro"):1.4f}')
    print(f'P_macro:\t{precision_score(trues, res, average="macro"):1.4f}')
    print(f'R_micro:\t{recall_score(trues, res, average="micro"):1.4f}')
    print(f'R_macro:\t{recall_score(trues, res, average="macro"):1.4f}')
    print(f'F1_micro:\t{f1_score(trues, res, average="micro"):1.4f}')
    print(f'F1_macro:\t{f1_score(trues, res, average="macro"):1.4f}')


def print_statistics_in_row(trues, res):
    print(f'A: {accuracy_score(trues, res):1.4f}', end=" | ")
    print(f'P_micro: {precision_score(trues, res, average="micro"):1.4f}', end=" | ")
    print(f'P_macro: {precision_score(trues, res, average="macro"):1.4f}', end=" | ")
    print(f'R_micro: {recall_score(trues, res, average="micro"):1.4f}', end=" | ")
    print(f'R_macro: {recall_score(trues, res, average="macro"):1.4f}', end=" | ")
    print(f'F1_micro: {f1_score(trues, res, average="micro"):1.4f}', end=" | ")
    print(f'F1_macro: {f1_score(trues, res, average="macro"):1.4f}')


def create_w2v_model(x_train: list):
    vector_size = 70
    window = 8
    sg = 1
    epochs = 15
    model = Word2Vec(x_train, vector_size=vector_size, window=window, sg=sg, epochs=epochs,
                     workers=4)
    return model


def group_samples_by_meaning(samples: list):
    res = defaultdict(list)
    for sample in samples:
        res[sample["meaning"]].append(sample["text"])
    return res


def create_vectorizor(x_train, model_type=None):
    # print("Model: Word2Vec + MeanEmbeddingVectorizer + StandardScaler")
    # model = Word2Vec(x_train, vector_size=70, window=8, sg=1, epochs=15, workers=4)
    # vectorizors = [MeanEmbeddingVectorizer(model)]
    # vectorizors.append(StandardScaler())

    # print("Model: BertTransformerEmbedding + cointegrated/rubert-tiny2")
    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    # cointegrated/rubert-tiny2
    # cointegrated/rubert-tiny
    model = BertTransformerEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorizors = [model]

    # print("Model: TfidfVectorizer")
    # model = TfidfVectorizer()
    # model.fit(x_train)
    # vectorizors = [model]

    return vectorizors


def full_classifier_list():
    return [
        (SVC(), f'SVM kernel=rbf'),
        (SVC(kernel='linear'), f'SVM linear'),
        (SVC(kernel='poly'), f'SVM kernel=poly'),
        (SVC(kernel='poly', coef0=0.75), f'SVM kernel=poly coef0=0.75'),
        (SVC(gamma=1), f'SVM kernel=rbf gamma=1'),
        (SVC(gamma=1, C=10), f'SVM kernel=rbf gamma=1 C=10'),
        (SVC(gamma=0.75, C=10), f'SVM kernel=rbf gamma=0.75 C=10'),
        (SVC(kernel='poly', degree=4, coef0=0.75), f'SVM kernel=poly degree=4 coef0=0.75'),
        (SVC(kernel='poly', degree=4, coef0=0.7), f'SVM kernel=poly degree=4 coef0=0.7'),
        (SVC(kernel='poly', degree=4, coef0=0.7, gamma=1, C=0.1),
                       f'SVM kernel=poly degree=4 coef0=0.7 gamma=1 C=0.1'),
         (SVC(degree=5, coef0=0.75, C=10), f'SVM kernel=rbf degree=5 coef0=0.75, C=10'),
         (SVC(degree=5, coef0=0.2), f'SVM kernel=rbf degree=5 coef0=0.2'),
         (SVC(kernel='poly', degree=5), f'SVM kernel=poly degree=5'),
         (SVC(kernel='poly', degree=5, coef0=0.65), f'SVM kernel=poly degree=5 coef0=0.65'),
         (SVC(kernel='poly', degree=5, coef0=0.75), f'SVM kernel=poly degree=5 coef0=0.75'),
         (SVC(kernel='poly', degree=5, coef0=0.7), f'SVM kernel=poly degree=5 coef0=0.7'),
         (SVC(kernel='poly', degree=6), f'SVM kernel=poly degree=6'),
         (SVC(kernel='poly', degree=6, coef0=0.75), f'SVM kernel=poly degree=6 coef0=0.75'),
         (SVC(kernel='poly', degree=6, coef0=0.7), f'SVM kernel=poly degree=6 coef0=0.7'),
         (SVC(class_weight='balanced', ), f'SVM class_weight=balanced kernel=rbf'),
         (SVC(class_weight='balanced', kernel='linear'), f'SVM class_weight=balanced linear'),
         (SVC(class_weight='balanced', kernel='poly'), f'SVM class_weight=balanced kernel=poly'),
         (SVC(class_weight='balanced', kernel='poly', coef0=0.75), f'SVM class_weight=balanced kernel=poly coef0=0.75'),
         (SVC(class_weight='balanced', gamma=1), f'SVM class_weight=balanced kernel=rbf gamma=1'),
         (SVC(class_weight='balanced', gamma=1, C=10), f'SVM class_weight=balanced kernel=rbf gamma=1 C=10'),
         (SVC(class_weight='balanced', gamma=0.75, C=10), f'SVM class_weight=balanced kernel=rbf gamma=0.75 C=10'),
         (SVC(class_weight='balanced', kernel='poly', degree=4, coef0=0.75),
          f'SVM class_weight=balanced kernel=poly degree=4 coef0=0.75'),
         (SVC(class_weight='balanced', kernel='poly', degree=4, coef0=0.7),
          f'SVM class_weight=balanced kernel=poly degree=4 coef0=0.7'),
         (SVC(class_weight='balanced', kernel='poly', degree=4, coef0=0.7, gamma=1, C=0.1),
                        f'SVM class_weight=balanced kernel=poly degree=4 coef0=0.7 gamma=1 C=0.1'),
          (SVC(class_weight='balanced', degree=5, coef0=0.75, C=10),
           f'SVM class_weight=balanced kernel=rbf degree=5 coef0=0.75, C=10'),
          (SVC(class_weight='balanced', degree=5, coef0=0.2),
           f'SVM class_weight=balanced kernel=rbf degree=5 coef0=0.2'),
          (SVC(class_weight='balanced', kernel='poly', degree=5),
           f'SVM class_weight=balanced kernel=poly degree=5'),
          (SVC(class_weight='balanced', kernel='poly', degree=5, coef0=0.65),
           f'SVM class_weight=balanced kernel=poly degree=5 coef0=0.65'),
          (SVC(class_weight='balanced', kernel='poly', degree=5, coef0=0.75),
           f'SVM class_weight=balanced kernel=poly degree=5 coef0=0.75'),
          (SVC(class_weight='balanced', kernel='poly', degree=5, coef0=0.7),
           f'SVM class_weight=balanced kernel=poly degree=5 coef0=0.7'),
          (SVC(class_weight='balanced', kernel='poly', degree=6),
           f'SVM class_weight=balanced kernel=poly degree=6'),
          (SVC(class_weight='balanced', kernel='poly', degree=6, coef0=0.75),
           f'SVM class_weight=balanced kernel=poly degree=6 coef0=0.75'),
          (SVC(class_weight='balanced', kernel='poly', degree=6, coef0=0.7),
           f'SVM class_weight=balanced kernel=poly degree=6 coef0=0.7'),
          (KNeighborsClassifier(), 'KNeighbors'),
          (KNeighborsClassifier(weights='distance'),
           'KNeighbors weights=distance'),
          (ExtraTreesClassifier(class_weight='balanced', n_estimators=500),
           f'ExtraTreesClassifier class_weight=balanced n_estimators=500'),
          (
              RandomForestClassifier(class_weight='balanced', ), f'RandomForest'),
          (
              RandomForestClassifier(bootstrap=False), f'RandomForest bootstrap=False'),
          (
              RandomForestClassifier(max_features=None), f'RandomForest max_features=None'),
          (
              RandomForestClassifier(criterion='entropy'), f'RandomForest entropy'),
          (
              RandomForestClassifier(criterion='entropy', max_features=None),
              f'RandomForest entropy max_features=None'),
          (
              RandomForestClassifier(criterion='entropy', max_features='log2'),
              f'RandomForest entropy max_features=log2'),
          (
              RandomForestClassifier(criterion='entropy', bootstrap=False),
              f'RandomForest entropy bootstrap=False'),
          (
              RandomForestClassifier(criterion='entropy', max_features=None, bootstrap=False),
              f'RandomForest entropy max_features=None bootstrap=False'),
          (
              RandomForestClassifier(criterion='entropy', max_features='log2', bootstrap=False),
              f'RandomForest entropy max_features=log2 bootstrap=False'),
          (
              RandomForestClassifier(n_estimators=150, criterion='entropy', bootstrap=False),
              f'RandomForest n_estimators=150 entropy bootstrap=False'),
          (
              RandomForestClassifier(n_estimators=200, criterion='entropy', bootstrap=False),
              f'RandomForest n_estimators=200 entropy bootstrap=False'),
          (
              RandomForestClassifier(n_estimators=500, criterion='entropy', bootstrap=False, max_features='log2'),
              f'RandomForest n_estimators=500 criterion=entropy bootstrap=False max_features=log2'
          )
          ]


def svm_teach_classify(filename):
    print("svm_teach_classify")
    print(filename)
    with open(f"./dicts/{filename}") as json_file:
        homonyms = json.load(json_file)
        best = []
        for classifier, name in full_classifier_list():
            micro_f_avg = 0
            macro_f_avg = 0
            print(name)
            for homonym, homonym_data in homonyms.items():
                samples = homonym_data["samples"]
                x, y = [], []
                test_size = 0.4
                for sample in samples:
                    y.append(sample["meaning"])
                    x.append(sample["text"])
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
                # print(f"test_size = {test_size} len x_train = {len(x_train)}, len x_test = {len(x_test)}, len y_train = {len(y_train)}, len y_test = {len(y_test)}")
                vectorizors = create_vectorizor(x_train)
                made_classifier = make_pipeline(*vectorizors, classifier)
                made_classifier.fit(x_train, y_train)
                y_res = made_classifier.predict(x_test)
                print(f"{homonym:10}", end=" | ")
                # print(name, end=" | ")
                print_statistics_in_row(y_test, y_res)
                micro_f_avg += f1_score(y_test, y_res, average="micro")
                macro_f_avg += f1_score(y_test, y_res, average="macro")
            micro_f_avg = micro_f_avg/len(list(homonyms.keys()))
            macro_f_avg = macro_f_avg/len(list(homonyms.keys()))
            best.append((name, micro_f_avg, macro_f_avg))
            print(f'F1_micro_avg: {micro_f_avg:1.4f}', end=" | ")
            print(f'F1_macro_avg: {macro_f_avg:1.4f}')
            print("____________")
        print("best:")
        best_res = max(best, key=itemgetter(1))
        print(f'{best_res[0]} | F1_micro_avg = {best_res[1]:1.4f} | F1_macro_avg = {best_res[2]:1.4f}')
        best_res = max(best, key=itemgetter(2))
        print(f'{best_res[0]} | F1_micro_avg = {best_res[1]:1.4f} | F1_macro_avg = {best_res[2]:1.4f}')

def main():
    warnings.filterwarnings('ignore')
    filename = "homonyms_with_50_samples.json"
    svm_teach_classify(filename)


if __name__ == "__main__":
    main()
