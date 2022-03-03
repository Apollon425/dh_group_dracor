import sys
import json
from urllib import request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd
import scipy
from dracor_nlp import Preprocessor

GER_METADATA_PATH = 'data_files/gerdracor-metadata.csv'
ITA_METADATA_PATH = 'data_files/itadracor-metadata.csv'
TF_IDF_PATH = f'data_files/ita_tfidf_min10.csv'



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
        texts.append(get_dracor(corpus, name, text_mode)) # Text herunterladen
        ids.append(ident)                                 # id hinzufügen
    return texts, ids                                     # Texte + ids als Ergebnis


# options: corpus="ita"/"ger", text="spoken"/"full"
def get_features(corpus="ita",
        text="full", 
        vocab=True, 
        syntax=True, 
        remove_stopwords=False, 
        lemmatize=True,
        normalize_orthography=False, 
        drama_stats=True, 
        get_ids=False

    ):
    features = []
    if corpus=="ita":                                      # Dramentexte aus dem Netz ziehen
        texts, ids = get_data("ita", text_mode=text)
        stopwordlist = stopwords.words('italian')
    elif corpus=="ger":
        texts, ids = get_data("ger", text_mode=text)
        stopwordlist = stopwords.words('german')
    else:
        print("No valid corpus found!")
        sys.exit()
    if not remove_stopwords:                               # Stopwordlisten deaktivieren falls gewünscht
        stopwordlist = None
    if lemmatize or sytax:
        preproc = Preprocessor(texts, corpus)
    if syntax:
        features.append(preproc.pos_tag())
    if lemmatize:
        texts = preproc.lemmatize()
    if vocab:  
        vectorizer = TfidfVectorizer(min_df=10, stop_words=stopwordlist, use_idf=True, norm=None)
        features.append(vectorizer.fit_transform(texts))
        if get_ids:
            features.append(ids, vectorizer.get_feature_names_out())
    return features


def convert_to_df_and_csv(path, scipymatrix, ids, export_data: bool) -> pd.DataFrame:
    df = pd.DataFrame(data=scipy.sparse.csr_matrix.todense(scipymatrix))
    df.columns = ids

    if export_data:
        write_to_csv(df, path, "utf-8", False)
    return df



def read_data_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def write_to_csv(data: pd.DataFrame, path: str, encoding: str, index: bool) -> None:
    data.to_csv(path, encoding=encoding, index=index)


if __name__ == '__main__':

    matrix, dracor_ids, vector_names = get_features("ita", get_ids= True)  #  do tf-idf
    df = convert_to_df_and_csv(TF_IDF_PATH, matrix, vector_names, True)  #  put data in pandas dataframe with named columns (=features), export as csv optionally
    df.index = dracor_ids  #  add row names (= dracors ids of plays)

    print(df)  #  df has named rows (=dracors ids of plays) and columns (feature)
