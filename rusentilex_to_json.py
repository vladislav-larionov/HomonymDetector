import csv
import json


def rusentilex_to_file():
    with open('rusentilex-2017.txt', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        with_definition = list(filter(lambda row: row['definition'], reader))
        definition_dict = dict()

        for word in with_definition:
            if word['word'] not in definition_dict:
                meanings = [{'word': word['word'],
                             "part": word['part'],
                             'lemma': word['lemma'],
                             "sentiment": word['sentiment'],
                             "source": word['source'],
                             'definition': word['definition'].replace(",", ", ").strip().lower()
                             }]
                definition_dict[word['word']] = {"meanings": meanings}
            else:
                meaning = {'word': word['word'],
                           "part": word['part'],
                           'lemma': word['lemma'],
                           "sentiment": word['sentiment'],
                           "source": word['source'],
                           'definition': word['definition'].replace(",", ", ").strip().lower()
                           }
                definition_dict[word['word']]["meanings"].append(meaning)
        with open('ambiguity_dict.json', 'w') as f:
            f.write(json.dumps(definition_dict, indent=4, ensure_ascii=False))


def join_rusentilex_json_with_samples():
    with open('ambiguity_dict.json') as json_file:
        ambiguity_dict = json.load(json_file)
        with open("corpora_nonempty.json") as corpora_nonempty_file:
            corpora_nonempty_dict = json.load(corpora_nonempty_file)
            ambiguity_samples = dict()
            for word in ambiguity_dict.keys():
                if word in corpora_nonempty_dict:
                    ambiguity_samples[word] = dict()
                    ambiguity_samples[word]["meanings"] = ambiguity_dict[word]["meanings"]
                    ambiguity_samples[word]["samples"] = corpora_nonempty_dict[word]
        with open('ambiguity_samples.json', 'w') as f:
            f.write(json.dumps(ambiguity_samples, indent=4, ensure_ascii=False))


def filter_rusentilex_json_by_sample_count(count):
    with open('ambiguity_samples.json') as ambiguity_samples_json:
        filtered_ambiguity_samples = dict()
        ambiguity_samples = json.load(ambiguity_samples_json)
        for word in ambiguity_samples.keys():
            if len(ambiguity_samples[word]["samples"]) >= count:
                filtered_ambiguity_samples[word] = dict()
                filtered_ambiguity_samples[word]["valid"] = True
                filtered_ambiguity_samples[word].update(ambiguity_samples[word])
                for sample in filtered_ambiguity_samples[word]["samples"]:
                    sample["valid"] = True
                    sample["text"] = sample["text"].replace("_", " ")
                for index, meaning in enumerate(filtered_ambiguity_samples[word]["meanings"]):
                    meaning["valid"] = True
                    meaning["index"] = index
        with open(f'ambiguity_filtered_by_{count}_samples.json', 'w') as f:
            f.write(json.dumps(filtered_ambiguity_samples, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    filter_rusentilex_json_by_sample_count(3)
