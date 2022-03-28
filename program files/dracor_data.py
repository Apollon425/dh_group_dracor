from email import header
from posixpath import split
import sys
import json
from urllib import request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd
import scipy
from dracor_nlp import Preprocessor
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns



GER_METADATA_PATH = Path("data_files/gerdracor-metadata.csv")
ITA_METADATA_PATH = Path("data_files/itadracor-metadata.csv")
OUTLIERLIST = ["ger000480"]  #  Karl Kraus - Die letzten Tage der Menschheit

#  metadata features to include ("id" must be present, rest optional):
metadata_featurelist = ["id", "yearNormalized", "numOfSpeakers", "numOfSpeakersFemale", "numOfSpeakersMale", "wordCountText", "wordCountSp", "wordCountStage"]


dracor_api = "https://dracor.org/api"                    # API-Endpunkt für DraCor

def get_dracor(corpus, play=None, text_mode=None):
    """Lädt Metadaten zum Korpus und den Text des Stücks."""
    url = dracor_api + "/corpora/" + corpus              # Basis-URL
    if text_mode==None or text_mode=="spoken":
        if play is not None:                             # Stück gewünscht?
            url = url + "/play/" + play + "/spoken-text" # URL für Text des Stückes
        with request.urlopen(url) as req:                # Daten herunterladen
            text = req.read().decode()                   # Daten einlesen
            if play is None:                             # Stück gewünscht?
                return json.loads(text)                  # JSON der Korpusmetadaten parsen und zurückgeben
            return text                                  # Text des Stückes zurückgeben
    elif text_mode=="full":
        url_sp = url + "/play/" + play + "/spoken-text"
        url_dr = url + "/play/" + play + "/stage-directions"
        with request.urlopen(url_sp) as req:
            text_spoken = req.read().decode()
        with request.urlopen(url_dr) as req:
            text_direct = req.read().decode()
        return " ".join([text_spoken, text_direct])
    else:
        print("No valid request for scope of text:")
        print(text_mode)
        print("supported options are: 'full', 'spoken' or None")
        sys.exit()  

def get_data(corpus, text_mode):
    """Alle Stücke eines Korpus herunterladen."""
    texts = []                                            # Texte der Stücke
    ids = []                                              # Autor*innen der Stücke
    for drama in get_dracor(corpus)["dramas"]:            # alle Stücke durchlaufen
        name = drama["name"]                              # Name des Stücks
        ident = drama["id"]                               # id des Stücks
        if ident in OUTLIERLIST:
            continue
        else:
            texts.append(get_dracor(corpus, name, text_mode)) # Text herunterladen
            ids.append(ident)                                 # id hinzufügen
    return texts, ids                                         # Texte + ids als Ergebnis

def get_metadata(corpus, ids: list):

    if corpus == "ger":
        meta = read_data_csv(GER_METADATA_PATH)
    elif corpus == "ita":
        meta = read_data_csv(ITA_METADATA_PATH)
    else:
        sys.exit("Corpus name invalid. Only \"ger\" and \"ita\" are supported.")

    #  sort dataframe to match the data downloaded via api:
    meta = meta[~meta['id'].isin(OUTLIERLIST)]  #  remove outliers from dataframe

    id_list = meta['id'].to_list()
    sort_index = []

    for dracor_id in id_list:
        sort_index.append(ids.index(dracor_id))

    meta['sort_index'] = sort_index
    meta.sort_values(by=["sort_index"], inplace=True)
    meta.drop(["sort_index"], axis=1, inplace=True)
    meta = meta.reset_index(drop=True)
    meta = meta[metadata_featurelist]

    return meta

def get_features(corpus="ita", text="full", syntax=True, remove_stopwords=False, lemmatize=True, min_df=10):  #  options: corpus="ita"/"ger", text="spoken"/"full"


    features = []
    if corpus=="ita":                                      # Dramentexte aus dem Netz ziehen
        texts, ids = get_data("ita", text_mode=text)
        stopwordlist = stopwords.words('italian')
    elif corpus=="ger":
        texts, ids = get_data("ger", text_mode=text)
        stopwordlist = stopwords.words('german')
    else:
        print("No valid corpus found!")
    if not remove_stopwords:                               # Stopwordlisten deaktivieren falls gewünscht
        stopwordlist = None
    if lemmatize or syntax:
        preproc = Preprocessor(texts, corpus)
    if syntax:
        features.append(preproc.pos_tag())
    if lemmatize:
        texts = preproc.lemmatize()
    vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stopwordlist, use_idf=True, norm=None)

    a = vectorizer.fit_transform(texts)
    features.append(a)  #  add tf-idf features
    features.append(ids)  #  add dracor ids
    features.append(vectorizer.get_feature_names_out())  #  add vector names for tf-idf vectors
    features.append(get_metadata(corpus, ids))  #  add metadata features

    return features


def convert_scipymatrix_to_dataframe(scipymatrix, ids) -> pd.DataFrame:
    """converts scipy matrix to pandas dataframe"""
    df = pd.DataFrame(data=scipy.sparse.csr_matrix.todense(scipymatrix))
    df.columns = ids
    return df


def dict_to_df(data: list) -> pd.DataFrame:
    """converts list of dictionarys to pandasdataframe"""
    df = pd.DataFrame(data)
    return df


def read_data_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def write_to_csv(data: pd.DataFrame, path: str, encoding: str, index: bool, header) -> None:
    data.to_csv(path, encoding=encoding, index=index, header=header)


if __name__ == '__main__':

    pass
    
    #ix, dracor_ids, vector_names,  meta_features = get_features("ger", vocab=True, syntax=True, get_ids= True, drama_stats=True)  #  do tf-idf
    # print(pos)
    # # for element in pos:
    # #     element
    # print(matrix)
    # print(dracor_ids)
    # print(vector_names)
    # print(meta_features)



