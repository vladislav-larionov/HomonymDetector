
# def is_ambiguity_valid(word_info):
#     valid = word_info["адекватность"]
#     return valid and len(list(filter(lambda sample: sample["адекватность"] and sample["meaning"] is not None,
#                                 word_info["samples"]))) > 2 and len(set(map(lambda sample: sample["meaning"], word_info["samples"]))) > 1

def is_ambiguity_valid(word_info):
    valid = True
    return valid and len(word_info["samples"]) > 2 and len(set(map(lambda sample: sample["meaning"], word_info["samples"]))) > 1


def read_and_filter_words(ambiguity_filtered_by_3_samples):
    return list(filter(lambda word: is_ambiguity_valid(ambiguity_filtered_by_3_samples[word]), ambiguity_filtered_by_3_samples.keys()))

