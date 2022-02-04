import nltk
from nltk.corpus import stopwords as sw
stopwords = sw.words("german")
if 'der' in stopwords:
    print("True")


import numpy as np
import json
import glob

#  Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#  spacy
import spacy
from spacy.lang.de.examples import sentences

#  visualization
import pyLDAvis
import pyLDAvis.gensim_models

def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def lemmatization(text):
    nlp = spacy.load("de_core_news_md")
    text = nlp.tokenizer(text)
    texts_out = []
    doc = nlp(text)
    final = " ".join([x.lemma_ for x in doc if x.text.lower() not in stopwords])
    texts_out.append(final)
    return (texts_out)

def gen_words(text):
    final = []
    new = gensim.utils.simple_preprocess(text)
    final.append(new)
    return final


#  preparing data



text_data = load_data("data_files/der_sterbende_cato.json")
#print(text_data)

lemmatized_text = lemmatization(text_data)
#print(lemmatized_text)

preprocessed_text = gen_words(lemmatized_text[0])
print(preprocessed_text)


id2word = gensim.corpora.Dictionary(preprocessed_text)

corpus = []
for word in preprocessed_text:
    new = id2word.doc2bow(word)
    corpus.append(new)
print(corpus[0])  #  tuple of index of word in dictionary and its frequency

print(id2word[[0][:1][0]])  #  get first element of first tuple of first text

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                            id2word = id2word, 
                                            num_topics=30, 
                                            random_state=100, 
                                            update_every=1, 
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto")


#  Visualization
#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R = 30)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')




