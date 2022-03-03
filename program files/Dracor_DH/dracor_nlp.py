import spacy

class Preprocessor:
    def __init__(self, texts, lang):
        self.spacy_docs = []
        if lang=="ita":
            nlp = spacy.load("it_core_news_lg", disable=['parser','ner'])
        else:
            nlp = spacy.load("de_dep_news_trf", disable=['parser'])
        for text in texts:
            self.spacy_docs.append(nlp(text))

    def __call__(self):     # __call__ and not a normal function name because tfidf-vectorizer demands a callable
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
