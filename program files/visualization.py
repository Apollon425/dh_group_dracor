from configparser import MissingSectionHeaderError
from turtle import shape

import sklearn
import dracor_data as dr
import pandas as pd
from matplotlib import pyplot as plt 

import matplotlib.cm as cm
import os


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




def cluster_scatterplot(
    
    top_terms: int, 
    corpus = "ger", 
    text ='spoken', 
    vocab = True,
    min_df=10, 
    syntax = False, 
    lemmatize = False, 
    get_ids = True, 
    label = None, 
    clusters = 6 ):


    """
    
    """

    #  1) get data:

    if corpus == "ger":
        meta = dr.read_data_csv(dr.GER_METADATA_PATH)
    elif corpus == "ita":
        meta = dr.read_data_csv(dr.ITA_METADATA_PATH)
    else:
        sys.exit("Corpus name invalid. Only \"ger\" and \"ita\" are supported.")


    return_list = dr.get_features(corpus=corpus,
                                 text=text, 
                                 vocab=vocab, 
                                 min_df=min_df, 
                                 syntax=syntax, 
                                 lemmatize=lemmatize, 
                                 get_ids=get_ids)

    matrix = return_list[0]
    dracor_ids = return_list[1]
    vector_names = return_list[2]

    df = dr.convert_to_df_and_csv(dr.TF_IDF_PATH, matrix, vector_names, False)  #  TODO: fix outputpath
    #print(df)



    #  2) cluster data using k-means:

    model = KMeans(n_clusters=clusters, init="k-means++", n_init=1, random_state=10).fit(df)  #  max_iter = 100

    # 3) dimension reduction of tf-idf vectors:

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
    df = df.drop(vector_names, axis=1)


    #  5) plot that df:

    plot = sns.relplot(data = df, x = 'x_axis', y = 'y_axis', hue = 'k_mean_cluster', palette = 'tab10', kind = 'scatter', height=15, aspect=1.5)
    if label is not None:
        ax = plot.axes[0, 0]
        for idx, row in df.iterrows():
            x = row[0]
            y = row[1]
            label_point_row = row[3]
            label_point = meta.loc[meta['id'] == f"{label_point_row}", f'{label}'].item()
            label_point = label_point + ", " + str((meta.loc[meta['id'] == f"{label_point_row}", f'yearNormalized'].item()))
            ax.text(x+25, y-10, label_point, horizontalalignment='left')


    #  6) save it:

    vocab_str = "tf_idf" if vocab is True else ""
    syntax_str = "pos" if syntax is True else ""
    lemma_str = "lemma" if lemmatize is True else ""
    label_str = label if label is not None else "None"


    out_string = f'/{corpus}_{text}_{vocab_str}_min_df={str(min_df)}_{syntax_str}_{lemma_str}_cluster={str(clusters)}_label={label_str}'  #  
    out_path = create_output_folder(out_string)
    if out_path != "":
        plt.savefig(out_path + "/cluster_plot.png")
        #plt.show()



    #  7) find contents of clusters, save them as csv

    #  metadata and token-list for plays in cluster:

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]  #  sort token by tf-idf value for each cluster
    #print(order_centroids)


    for cluster in range(clusters):
        cluster_content = df.query(f'k_mean_cluster=={cluster}')['dracor_id'].to_list()

        meta_data_cluster = meta.loc[meta['id'].isin(cluster_content)]
        print(f"\n Metadata for Cluster {cluster}: \n\n {meta_data_cluster} \n ------------------- \n")
        if out_path != "":
            meta_data_cluster.to_csv(out_path + f"/cluster {cluster}.csv")

        #   TODO:  fix output of highest tf-idf features per cluster
        #  
        # highest_tf_idf_scores = order_centroids[cluster][:top_terms]
        # print(highest_tf_idf_scores)
        # for value in highest_tf_idf_scores:
        #     try:
        #         print(df_2.columns[value])
        #     except IndexError as e:
        #         print("Fehler bei: \n")

        #         print(value)

    #  token-list for plays in cluster:


    # with open ("results2.txt", "w", encoding="utf-8") as f:
    #     for i in range(clusters):
    #         f.write(f"Cluster: {i} \n")
    #         #f.write("\n")
    #         for ind in order_centroids[i, :top_terms]:
    #             f.write(' %s' % vector_names[ind],)
    #             f.write("\n")
    #         f.write("\n")
    #         f.write("\n")



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

    cluster_scatterplot(top_terms=25, 
                      corpus="ita", 
                      text='full', 
                      vocab=True, 
                      min_df=10,
                      syntax=False, 
                      lemmatize=True, 
                      get_ids=True,
                      #label='firstAuthor',
                      clusters=6)

