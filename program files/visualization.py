from configparser import MissingSectionHeaderError
from turtle import shape

import sklearn
import dracor_data as dr
import pandas as pd
from matplotlib import pyplot as plt 

import matplotlib.cm as cm
import os
from pathlib import Path

import sys
import elbow as elb
import silhouette as sil


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
import dracor_data as dr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import seaborn as sns

OUTPUT_PATH_BASE = 'visualization_output/clustering'
out_path = ""

def create_output_folder(output_folder: str):

    outputpath = OUTPUT_PATH_BASE + output_folder
    path_exists = os.path.exists(outputpath)

    if not path_exists:    
        os.makedirs(outputpath)
        print("New plot saved.")
        return outputpath
    else:
        print("Plot not saved, since one with the same parameters already exists.")
        return ""
        



def set_time_frame(first_year, last_year, corpus) -> tuple:

    """help function for draw_plot, determines time frame to draw plot for; if not specified by user, it defaults to the oldest/youngest play in the corpus 
    """
    path = dr.GER_METADATA_PATH if corpus == "ger" else dr.ITA_METADATA_PATH

    if not first_year:
        first_year = dr.read_data_csv(path)['yearNormalized'].min()
    if not last_year:
        last_year = dr.read_data_csv(path)['yearNormalized'].max()


    return first_year, last_year




def draw_plot(data: pd.DataFrame, column: str, plot_type: str, annotate: bool, first_year: int = None, last_year: int = None, corpus="ger") -> None:

    """
    :param column: y-axis value to plot against time; must be column name out of given data frame (see parameter data')
    :param plot_type: name of plot-type; currently only supports scatter-plot
    :param annotate: if True, data points are labeled (currently with name of the author)
    """

    first_year, last_year = set_time_frame(first_year, last_year, corpus)
    data = data.loc[(data['yearNormalized'] >= first_year) & (data['yearNormalized'] <= last_year)]  #  select only given time frame

    if plot_type == "scatter":  #  TODO: dictionary containing different plot-types
        plt.scatter(data["yearNormalized"], data[column])
        plt.ylabel(column)



    if annotate:
        year = data["yearNormalized"].to_list()
        y_axis_attribute = data[column].to_list()
        name = data["firstAuthor"].to_list()

        for i, txt in enumerate(name):
            plt.annotate(txt, (year[i], y_axis_attribute[i]))

    plt.show()


def write_centroids(clusters, order_centroids, top_terms, vector_names):

    file_path = Path(out_path + f"/Top_{top_terms}_centroids.txt")
    open(file_path, mode='a').close()
    
    with open (file_path, "w", encoding="utf-8") as f:
        for i in range(clusters):
            f.write(f"Cluster: {i} \n")
            print(f"Cluster: {i} \n")
            f.write("\n")
            for token_index in order_centroids[i][:top_terms]:
                f.write(f'{vector_names[token_index]}')
                print(vector_names[token_index])
                f.write("\n")
            f.write("\n")
            f.write("\n")


def cluster_scatterplot(
    
    top_terms: int, 
    corpus = "ger", 
    text ='spoken', 
    vocab = True,
    min_df=10,
    removeStopwords = False, 
    syntax = False, 
    lemmatize = False, 
    get_ids = True, 
    label = None,
    drama_stats = False, 
    clusters = 6):


    """
    
    """

    #  1) get data:

    if corpus == "ger":
        meta = dr.read_data_csv(dr.GER_METADATA_PATH)
    elif corpus == "ita":
        meta = dr.read_data_csv(dr.ITA_METADATA_PATH)
    else:
        sys.exit("Corpus name invalid. Only \"ger\" and \"ita\" are supported.")


    pos, matrix, dracor_ids, vector_names,  meta_features = dr.get_features(corpus=corpus,
                                                                            text=text,
                                                                            remove_stopwords=removeStopwords, 
                                                                            vocab=vocab, 
                                                                            min_df=min_df, 
                                                                            syntax=syntax, 
                                                                            lemmatize=lemmatize, 
                                                                            get_ids=get_ids)


  
    df = dr.convert_to_df_and_csv(dr.TF_IDF_PATH, matrix, vector_names, False)  #  TODO: fix outputpath
    #print(df)
    #print(meta_features)
    print("df mit meta sorted right:")
    df = pd.concat([df, meta_features], axis=1)
    #print("dracor ids:")
    #print(dracor_ids)
    #print(pos)
    key_list = list(pos[0].keys())
    print("key list:")
    print(key_list)
    pos_df = dr.dict_to_df(pos)
   # print("pos_df")
    df = pd.concat([df, pos_df], axis=1)
    df = df.drop(['id'], axis=1)
    print("final df:")
    print(df)


    #  2) cluster data using k-means:

    model = KMeans(n_clusters=clusters, init="k-means++", n_init=1, random_state=10).fit(df)  #  max_iter = 100

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]  #  sort token by tf-idf value for each cluster; TODO: still work after meta+pos features?


    # 3) dimension reduction feature vectors:

    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(matrix.toarray())

    #  4) build new df for more convenient plotting:

    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]
    df['x_axis'] = x_axis
    df['y_axis'] = y_axis
    kmean_indices = model.fit_predict(df)
    df['k_mean_cluster'] = kmean_indices
    df['dracor_id'] = dracor_ids
    df = df.drop(vector_names, axis=1)  #  TODO: why drop vector names? drop all columns? how to drop
    df = df.drop(dr.metadata_featurelist[1:], axis=1)  #  except first element (id), because it has been dropped earlier; 'id' can be removed from dr.metadate_feature_list if correctness of df has been confirmed
    df = df.drop(key_list, axis=1)
    print("df after clustering:")
    print(df)


    #  5) plot that df:

    plot = sns.relplot(data = df, x = 'x_axis', y = 'y_axis', hue = 'k_mean_cluster', palette = 'tab10', kind = 'scatter', height=15, aspect=1.5)
    if label is not None:
        ax = plot.axes[0, 0]
        for idx, row in df.iterrows():
            x = row[0]
            y = row[1]
            label_point_row = row[3]
            print("label point row:")
            print(label_point_row)
            label_point = meta.loc[meta['id'] == f"{label_point_row}", f'{label}'].item()
            print(f"label point {idx}:")
            print(label_point)
            label_point = label_point + ", " + str((meta.loc[meta['id'] == f"{label_point_row}", f'yearNormalized'].item()))
            ax.text(x+25, y-10, label_point, horizontalalignment='left')


    #  6) save it:

    vocab_str = "tf_idf" if vocab is True else ""
    syntax_str = "pos" if syntax is True else ""
    lemma_str = "lemma" if lemmatize is True else ""
    label_str = "yes" if label is not None else "no"
    stopword_str = "noStop" if removeStopwords is True else ""


    out_string = f'/{corpus}_{text}_{stopword_str}_{vocab_str}_min_df={str(min_df)}_{syntax_str}_{lemma_str}_cluster={str(clusters)}_lab={label_str}'   
    out_path = create_output_folder(out_string)
    if out_path != "":
        plt.savefig(out_path + "/cluster_plot.png")
        #plt.show()



    #  7) find contents of clusters, save them as csv

    #  token-list and metadata for plays in cluster:
    
    try:
        write_centroids(clusters=clusters, order_centroids=order_centroids, top_terms=top_terms, vector_names=vector_names)
    except (FileNotFoundError, PermissionError) as e:
        print(e)
        print("Error writing centroids.")

    
    for cluster in range(clusters):
        cluster_content = df(f'k_mean_cluster=={cluster}')['dracor_id'].to_list()

        meta_data_cluster = meta.loc[meta['id'].isin(cluster_content)]
        print(f"\n Metadata for Cluster {cluster}: \n\n {meta_data_cluster} \n ------------------- \n")
        if out_path != "":
            meta_data_cluster.to_csv(out_path + f"/cluster {cluster}.csv")


if __name__ == '__main__':

    #  metadata plotting:

    #  draw_plot(data=read_data_csv(dr.GER_METADATA_PATH), column="averageClustering", plot_type="scatter", annotate=False, first_year=1850)

    #  calculate tif-idf (+ get singular token if that is of interest):

    # return_list = dr.get_features("ita", text='spoken', vocab=True, syntax=False, lemmatize=False, get_ids= True)
    # matrix = return_list[0]
    # dracor_ids = return_list[1]
    # vector_names = return_list[2]

    # print(dracor_ids)
    # print(dracor_ids[138])
    # print(vector_names)
    # print(vector_names[1120])


    #  elbow/silhouette:

    # #elb.elbow_plot(data=matrix, no_of_clusters=5)
    # sil.silhouette_plot(data=matrix, no_of_clusters=5)


    #  cluster plotting:

    cluster_scatterplot(top_terms=10, 
                      corpus="ita", 
                      text='spoken', 
                      vocab=True, 
                      min_df=20,
                      removeStopwords=True,
                      syntax=True, 
                      lemmatize=True, 
                      get_ids=True,
                      label='firstAuthor',
                      drama_stats=True,
                      clusters=15)

