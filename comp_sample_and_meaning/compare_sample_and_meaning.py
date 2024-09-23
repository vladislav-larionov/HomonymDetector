from comp_sample_and_meaning.bertscore import bert_score
from comp_sample_and_meaning.d2v_emb import d2v_emb
from comp_sample_and_meaning.gensim_pretrainde import gensim_pretrainde
from comp_sample_and_meaning.navec_emb import navec_score
from comp_sample_and_meaning.w2v_emb import w2v_emb


def main():
    filename = "homonyms_with_50_samples.json"
    # filename = "narusco_ru.json"
    # filename = "homonyms_ru.json"
    w2v_emb(filename)
    navec_score(filename)
    gensim_pretrainde(filename)
    d2v_emb(filename)
    bert_score(filename)


if __name__ == "__main__":
    main()
