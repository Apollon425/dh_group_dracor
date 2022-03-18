import spacy

taglist = ["ADJ","ADP","ADV","AUX","CONJ","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X","SPACE"]

class Preprocessor:
    def __init__(self, texts, lang):
        self.spacy_docs = []
        if lang=="ita":
            nlp = spacy.load("it_core_news_lg", disable=['parser','ner'])
        else:
            nlp = spacy.load("de_core_news_sm", disable=['parser'])
            nlp.max_length = 5000000
        for doc in nlp.pipe(texts, disable=["parser", "attribute_ruler", "ner"]):  #  "tok2vec", 
            self.spacy_docs.append(doc)  #  nlp (doc) ??

    def lemmatize(self):
        lemmatized_texts = []
        for doc in self.spacy_docs:
            lemmatized_texts.append(" ".join([token.lemma_ for token in doc]))
        return lemmatized_texts

    def pos_tag(self, relative=True):
        pos_tag_features = []
        for doc in self.spacy_docs:
            pos=dict()
            # text_pos = [token.pos_ for token in doc]    # POS bigrams?
            num_pos = doc.count_by(spacy.attrs.POS)
            for k,v in sorted(num_pos.items()):
                pos.update({doc.vocab[k].text:v})
            for tag in taglist:
                if tag not in pos.keys():
                    pos.update({tag:0})
            if relative:                           # relative count of POS
                for keytag in pos.keys():
                    pos.update(keytag:(pos[keytag]/sum(pos.values()))
            pos_tag_features.append(pos)
        return pos_tag_features