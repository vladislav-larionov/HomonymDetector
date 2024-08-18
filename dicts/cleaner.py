import json
import sys


def remove_samples_with_no_meaning(hom_dictionary: dict):
    cleaned = dict()
    for word, word_data in hom_dictionary.items():
        cleaned_word = dict(**word_data)
        cleaned_word["samples"] = []
        for sample in word_data["samples"]:
            if sample["meaning"] is not None:
                cleaned_word["samples"].append(sample)
        if len(cleaned_word["samples"]):
            cleaned[word] = cleaned_word
    return cleaned


def remove_meaning_with_no_samples(hom_dictionary: dict):
    cleaned = dict()
    for word, word_data in hom_dictionary.items():
        cleaned_word = dict(**word_data)
        cleaned_word["meanings"] = []
        for meaning in word_data["meanings"]:
            if len(list(filter(lambda sample: sample["meaning"] == meaning["index"], cleaned_word["samples"]))) > 0:
                cleaned_word["meanings"].append(meaning)
        if len(cleaned_word["meanings"]) > 1:
            cleaned[word] = cleaned_word
    return cleaned


def main():
    print(f"file: {sys.argv[0]}")
    filename = sys.argv[1]
    with open(filename) as source_file:
        hom_dictionary = json.load(source_file)
        print(f"remove_samples_with_no_meaning")
        filtered_homs = remove_samples_with_no_meaning(hom_dictionary)
        print(f"remove_meaning_with_no_samples")
        filtered_homs = remove_meaning_with_no_samples(filtered_homs)

        with open(f'{filename.replace(".json", "")}_cleaned.json', 'w') as f:
            f.write(json.dumps(filtered_homs, indent=4, ensure_ascii=False))



if __name__ == "__main__":
    main()