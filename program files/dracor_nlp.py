import spacy
from collections import Counter


class Preprocessor:
    def __init__(self, texts, lang):
        self.spacy_docs = []
        if lang=="ita":
            nlp = spacy.load("it_core_news_lg", disable=['parser', "ner"])  #  ,'ner'
        else:
            nlp = spacy.load("de_core_news_sm", disable=['parser'])
            nlp.max_length = 5000000
        for doc in nlp.pipe(texts, disable=["parser", "attribute_ruler", "ner"]):  #  "tok2vec", "parser", "attribute_ruler" -- spaces, "ner" "tok2vec", "tagger"
            self.spacy_docs.append(doc)  #  nlp (doc) ??

    def lemmatize(self):
        lemmatized_texts = []
        for doc in self.spacy_docs:
            lemmatized_texts.append(" ".join([token.lemma_ for token in doc]))
        return lemmatized_texts

    # def pos_tag(self):
    #     pos_tag_features = []
    #     for doc in self.spacy_docs:
    #         #print("doc:")
    #         #print(doc)
    #         pos=dict()
    #         # text_pos = [token.pos_ for token in doc]    # POS bigrams?
    #         print(spacy.attrs.POS)
    #         num_pos = doc.count_by(spacy.attrs.POS)
    #         print("num_pos")
    #         print(num_pos)
    #         for k,v in sorted(num_pos.items()):
    #             pos.update({doc.vocab[k].text:v})
    #         pos_tag_features.append(pos)

    #     return pos_tag_features



# sbase = sum(c.values())
# for el, cnt in c.items():
#     print(el, '{0:2.2f}%'.format((100.0* cnt)/sbase))







    def pos_tag(self):
        pos_tag_features = []
        for doc in self.spacy_docs:
            c = Counter(([token.pos_ for token in doc]))
            sbase = sum(c.values())
            print(c.values())

            for el, cnt in c.items():
                print(el, '{0:2.2f}%'.format((100.0* cnt)/sbase))
            #print("doc:")
            #print(doc)
            pos=dict()
            # text_pos = [token.pos_ for token in doc]    # POS bigrams?
            #print(spacy.attrs.POS)
            #num_pos = doc.count_by(spacy.attrs.POS)
            #print("num_pos")
            #print(num_pos)
            #for k,v in sorted(num_pos.items()):
            #    pos.update({doc.vocab[k].text:v})
            #pos_tag_features.append(pos)

        #return pos_tag_features
        return pos