import re

import requests
import urllib
from bs4 import BeautifulSoup
import json

def clean_up_text(text):
    text = text.replace("+", "").replace("...", "").replace("^", "").replace("--", "-").replace("\r\n", "\n").replace("\r\n \n", "\n").replace("_", "-").strip()
    text = re.sub("\s+", " ", text)
    return text


def parse_texts_for_word(word) -> list:
    encoded_word = urllib.parse.quote(word, safe='/', encoding='cp1251', errors=None)
    response = requests.get(f"https://narusco.ru/search/ac1.php?wf={encoded_word}&yo=on&compounds=on").text
    bs = BeautifulSoup(response, 'html.parser')
    texts = []
    cur_text = ""
    cur_title = ""
    for p_tag in bs.find_all('p', {"class": ["resHeader", "res", "delim"]}):
        if p_tag['class'][0] == "resHeader":
            cur_title = p_tag.text
            if cur_text:
                texts.append({"text": clean_up_text(cur_text), "title": clean_up_text(cur_title), "meaning": None})
            cur_text = ""
        if p_tag['class'][0] == "res":
            cur_text += p_tag.text + " "
        if p_tag['class'][0] == "delim":
            cur_text += "\n"
    if cur_text:
        texts.append({"text": clean_up_text(cur_text), "title": clean_up_text(cur_title), "meaning": None})
    return texts


def save_to_json_file(filename, corpora):
    with open(f"{filename}.json", "w") as f:
        f.write(json.dumps(corpora, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    with open('ambiguity_dict.json') as json_file:
        ambiguity_dict = json.load(json_file)
        corpora_all = dict()
        corpora_nonempty = dict()
        for word in list(ambiguity_dict.keys()):
            corpora_item = {word: parse_texts_for_word(word)}
            corpora_all.update(corpora_item)
            if len(corpora_item[word]):
                corpora_nonempty.update(corpora_item)
            print(corpora_item)
        save_to_json_file("corpora_all", corpora_all)
        save_to_json_file("corpora_nonempty", corpora_nonempty)
