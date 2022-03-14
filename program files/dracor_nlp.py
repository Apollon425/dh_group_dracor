import spacy

class Preprocessor:
    def __init__(self, texts, lang):
        self.spacy_docs = []
        if lang=="ita":
            nlp = spacy.load("it_core_news_lg", disable=['parser','ner'])
        else:
            nlp = spacy.load("de_core_news_sm", disable=['parser'])
            nlp.max_length = 5000000
        for doc in nlp.pipe(texts, disable=["tok2vec", "parser", "attribute_ruler", "ner"]):
            self.spacy_docs.append(doc)  #  nlp (doc) ??

    def lemmatize(self):
        lemmatized_texts = []
        for doc in self.spacy_docs:
            lemmatized_texts.append(" ".join([token.lemma_ for token in doc]))
        return lemmatized_texts

    def pos_tag(self):
        pos_tag_features = []
        for doc in self.spacy_docs:
            # text_pos = [token.pos_ for token in doc]    # POS bigrams?
            pos_tag_features.append(doc.count_by(spacy.attrs.POS))  # convert to panda df?
        return pos_tag_features