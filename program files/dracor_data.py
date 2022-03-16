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

# GER_METADATA_PATH = 'data_files/gerdracor-metadata.csv'
# ITA_METADATA_PATH = 'data_files/itadracor-metadata.csv'
# TF_IDF_PATH = f'data_files/ita_tfidf_min10.csv'

GER_METADATA_PATH = Path("data_files/gerdracor-metadata.csv")
ITA_METADATA_PATH = Path("data_files/itadracor-metadata.csv")
TF_IDF_PATH = Path("data_files/ita_tfidf_min10.csv")

#EXCLUDE_PLAYS = ["ger000480"]
EXCLUDE_PLAYS = ["ita000001", "ita000002", "ita000003", "ita000004", "ita000005", "ita000006", "ita000007", "ita000008", "ita000009", "ita000010", "ita000011", "ita000012",
 "ita000013", "ita000014", "ita000015", "ita000016", "ita000017", "ita000018", "ita000019", "ita000020", "ita000021", "ita000022", "ita000023", "ita000024", "ita000025",
  "ita000026", "ita000027", "ita000028", "ita000029", "ita000030", "ita000031", "ita000032", "ita000033", "ita000034", "ita000035", "ita000036", "ita000037", "ita000038",
   "ita000039", "ita000040", "ita000041", "ita000042", "ita000043", "ita000044", "ita000045", "ita000046", "ita000047", "ita000048", "ita000049", "ita000050", "ita000051",
    "ita000052", "ita000053", "ita000054", "ita000055", "ita000056", "ita000057", "ita000058", "ita000059", "ita000060", "ita000061", "ita000062", "ita000063", "ita000064",
    "ita000065", "ita000066", "ita000067", "ita000068", "ita000069", "ita000070"
 ]
#      , "ita000071", "ita000072", "ita000073", "ita000074", "ita000075", "ita000076", "ita000077",
#       "ita000078", "ita000079", "ita000080", "ita000081", "ita000082", "ita000083", "ita000084", "ita000085", "ita000086", "ita000087", "ita000088", "ita000089", "ita000090",
#        "ita000091", "ita000092", "ita000093", "ita000094", "ita000095", "ita000096", "ita000097", "ita000098", "ita000099", "ita000100", "ita000101", "ita000102", "ita000103",
#         "ita000104", "ita000105", "ita000106", "ita000107", "ita000108", "ita000109", "ita000110", "ita000111", "ita000112", "ita000113", "ita000114", "ita000115", "ita000116",
#          "ita000117", "ita000118", "ita000119", "ita000120"
#  , "ita000121", "ita000122", "ita000123", "ita000124", "ita000125", "ita000126", "ita000127", "ita000128", "ita000129",
#           "ita000130", "ita000131", "ita000132", "ita000133", "ita000134", "ita000135"


metadata_featurelist = ["yearNormalized", "numOfSpeakers", "numOfSpeakersFemale", "numOfSpeakersMale", "wordCountText", "wordCountSp", "wordCountStage"]




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
        if ident in EXCLUDE_PLAYS:
            continue
        else:
            texts.append(get_dracor(corpus, name, text_mode)) # Text herunterladen
            ids.append(ident)                                 # id hinzufügen
    #print(ids)
    return texts, ids                                         # Texte + ids als Ergebnis

def get_metadata(corpus, ids: list):
    if corpus == "ger":
        meta = read_data_csv(GER_METADATA_PATH)
    elif corpus == "ita":
        meta = read_data_csv(ITA_METADATA_PATH)
    else:
        sys.exit("Corpus name invalid. Only \"ger\" and \"ita\" are supported.")

    #  sort dataframe to match the data downloaded via api:
    meta = meta[~meta['id'].isin(EXCLUDE_PLAYS)]
    id_list = meta['id'].to_list()
    sort_index = []

    for dracor_id in id_list:
        sort_index.append(ids.index(dracor_id))

    meta['sort_index'] = sort_index
    meta.sort_values(by=["sort_index"], inplace=True)
    meta.drop(["sort_index"], axis=1, inplace=True)

    meta = meta[metadata_featurelist]
    #print(meta)

    return meta

# options: corpus="ita"/"ger", text="spoken"/"full"
def get_features(corpus="ita",
        text="full", 
        vocab=True, 
        syntax=True, 
        remove_stopwords=False, 
        lemmatize=True, 
        drama_stats=True, 
        get_ids=False,
        min_df=10

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
    if not remove_stopwords:                               # Stopwordlisten deaktivieren falls gewünscht
        stopwordlist = None
    if lemmatize or syntax:
        preproc = Preprocessor(texts, corpus)
    if syntax:
        features.append(preproc.pos_tag())
    if lemmatize:
        texts = preproc.lemmatize()
    if vocab:  
        vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stopwordlist, use_idf=True, norm=None)
        features.append(vectorizer.fit_transform(texts))
        if get_ids:
            features.append(ids)
            features.append(vectorizer.get_feature_names_out())
    if drama_stats:
        features.append(get_metadata(corpus, ids))
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


def create_sublists(corpus: str, split_in: int) -> list:
    sublists = []
    if corpus == "ita":

        no_plays = len(read_data_csv(ITA_METADATA_PATH).index)

        split_indices = ([no_plays // split_in + (1 if x < no_plays % split_in else 0)  for x in range (split_in)])  #  split in equal parts

        stop_indices = get_stop_indices(split_list=split_indices)

        for i in range(0, split_in):
            sublist = []
            for x in range(stop_indices[i], stop_indices[i+1]):
                #print(stop_indices[i])
                #print(stop_indices[i+1])

                
                id = corpus + "{:06d}".format(x+1)
                #print(id)
                sublist.append(id)
            #print(f"sublist {i+1}: ")
            #print(sublist)
            #print(len(sublist))
            sublists.append(sublist)

    return sublists





def get_stop_indices(split_list: list) -> list:
    """help function for create sublist, returns indices in a list to build a sublist of dracor_ids to use"""
    new_list = []
    for i in range(0, len(split_list)):

        if i == 0:
            new_list.append(0)
            new_list.append(split_list[i])
        else:
            this_value = new_list[-1] + split_list[i]

            new_list.append(this_value)
    return new_list


    # foo = somevalue
    # previous = next_ = None
    # l = len(objects)
    # for index, obj in enumerate(objects):
    #     if obj == foo:
    #         if index > 0:
    #             previous = objects[index - 1]
    #         if index < (l - 1):
    #             next_ = objects[index + 1]

    


if __name__ == '__main__':

    sublists = create_sublists("ita", 3)
    for sl in sublists:
        EXCLUDE_PLAYS = sl
        pos, matrix, dracor_ids, vector_names,  meta_features = get_features("ita", vocab=True, get_ids= True, drama_stats=True)  #  do tf-idf
        print(pos)



    # pos, matrix, dracor_ids, vector_names,  meta_features = get_features("ita", vocab=True, get_ids= True, drama_stats=True)  #  do tf-idf
    # print(pos)
    # for element in pos:
    #     element
    #print(matrix)
    #print(dracor_ids)
    #print(vector_names)
    #print(meta_features)



    #df = convert_to_df_and_csv(TF_IDF_PATH, matrix, vector_names, True)  #  put data in pandas dataframe with named columns (=features), export as csv optionally
    #df.index = dracor_ids  #  add row names (= dracors ids of plays)
    #print(df)  #  df has named rows (=dracors ids of plays) and columns (feature)


    

    
