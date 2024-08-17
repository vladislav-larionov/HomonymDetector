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
    response = requests.get(f"https://homonyms.ru/search?q={word}").text
    bs = BeautifulSoup(response, 'html.parser')
    res = dict()
    if not bs.find("main").find("p", class_="pages"):
        response = requests.get(f'https://homonyms.ru{bs.find("main").find("a")["href"]}').text
        bs = BeautifulSoup(response, 'html.parser')
        word_dict = dict(адекватность=True, meanings=[], samples=[])
        for word_tag in bs.find_all("dd"):
            defin = word_tag.text
            for sample in word_tag.find_all("q"):
                print(f"word = {word}\nsample = {sample.text}")
                defin = defin.replace(sample.text, "")
                word_dict["samples"].append({"text": sample.text, "адекватность": True, "meaning": len(word_dict["meanings"])})
            defin = re.sub(r"[\s]{2,}", ', ', defin.strip())
            word_dict["meanings"].append({"index": len(word_dict["meanings"]), "определение": defin, "адекватность": True})
            print(f"defin = {defin}")
        res.update(word_dict)
    return res

def save_to_json_file(filename, corpora):
    with open(f"{filename}.json", "w") as f:
        f.write(json.dumps(corpora, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    with open('ambiguity_dict.json') as json_file:
        ambiguity_dict = json.load(json_file)
        corpora_nonempty = dict()
        for word in list(ambiguity_dict.keys()):
            corpora_item = parse_texts_for_word(word)
            if corpora_item:
                corpora_nonempty.update({word: parse_texts_for_word(word)})
        save_to_json_file("homonyms_ru", corpora_nonempty)
