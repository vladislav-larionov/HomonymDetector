from comp_sample_and_meaning.bertscore import bert_score
from comp_sample_and_meaning.d2v_emb import d2v_emb
from comp_sample_and_meaning.gensim_pretrainde import gensim_pretrainde
from comp_sample_and_meaning.navec_emb import navec_score
from comp_sample_and_meaning.w2v_emb import w2v_emb


def main():
    # filename = "homonyms_with_50_samples.json"
    # filename = "narusco_ru.json"
    filename = "homonyms_ru.json"
    filename = "homonyms_ru_clean.json"
    with open("../results/comp_sample_and_meaning/res_total_corpora.md", "w") as file:
        print(f"# {__file__}\n", file=file)
        for filename in ["corpora.json"]:
            print(f"Корпус {filename}\n", file=file)
            w2v_emb(filename, file=file)
            navec_score(filename, file=file)
            gensim_pretrainde(filename, file=file)
            d2v_emb(filename, file=file)
            bert_score(filename, file=file)


if __name__ == "__main__":
    main()
