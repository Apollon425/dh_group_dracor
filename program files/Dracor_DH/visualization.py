from configparser import MissingSectionHeaderError
from turtle import shape
import dracor_data as dr
import pandas as pd
from matplotlib import pyplot as plt
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




def set_time_frame(first_year, last_year) -> tuple:

    """help function for draw_plot, determines time frame to draw plot for; if not specified by user, it defaults to the oldest/youngest play in the corpus 
    """

    if not first_year:
        first_year = dr.read_data_csv(dr.METADATA_PATH)['yearNormalized'].min()
    if not last_year:
        last_year = dr.read_data_csv(dr.METADATA_PATH)['yearNormalized'].max()

    return first_year, last_year

    

def draw_plot(data: pd.DataFrame, column: str, plot_type: str, annotate: bool, first_year: int = None, last_year: int = None) -> None:

    """
    :param column: y-axis value to plot against time; must be column name out of given data frame (see parameter data')
    :param plot_type: name of plot-type; currently only supports scatter-plot
    :param annotate: if True, data points are labeled (currently with name of the author)
    """

    first_year, last_year = set_time_frame(first_year, last_year)
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

def cluster_scatterplot(top_termns: int, corpus= "ger"):

    if corpus == "ger":
        meta = dr.read_data_csv(dr.GER_METADATA_PATH)
    elif corpus == "ita":
        meta = dr.read_data_csv(dr.ITA_METADATA_PATH)
    else:
        sys.exit("Corpus name invalid. Only \"ger\" and \"ita\" are supported.")


    return_list = dr.get_features(corpus=corpus, text='spoken', vocab=True, syntax=False, lemmatize=False, get_ids= True)
    matrix = return_list[0]
    dracor_ids = return_list[1]
    vector_names = return_list[2]



    names = meta['firstAuthor'].to_list()


    k = 6
    model = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=10)  #  max_iter = 100

    model.fit(matrix)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    print(order_centroids)

    with open ("results.txt", "w", encoding="utf-8") as f:
        for i in range(k):
            f.write(f"Cluster: {i} \n")
            #f.write("\n")
            for ind in order_centroids[i, :top_termns]:
                f.write(' %s' % vector_names[ind],)
                f.write("\n")
            f.write("\n")
            f.write("\n")



    kmean_indices = model.fit_predict(matrix)
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(matrix.toarray())


    colors = ["r", "b", "c", "y", "m", "g"]  #  "k", "yellow", "greenyellow", "pink"

    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]

    fix, ax = plt.subplots(figsize=(50, 50))

    ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

    for i, txt in enumerate(names):
        ax.annotate(txt[0:5], (x_axis[i], y_axis[i]))

    plt.savefig("test3.png")

if __name__ == '__main__':

    # #  metadata plotting:
    # #draw_plot(data=read_data_csv(METADATA_PATH), column="averageClustering", plot_type="scatter", annotate=False, first_year=1850)

    # #  calculate tif-idf:
    # return_list = dr.get_features("ita", text='spoken', vocab=True, syntax=False, lemmatize=False, get_ids= True)
    # matrix = return_list[0]
    # dracor_ids = return_list[1]
    # vector_names = return_list[2]


    # print(dracor_ids)
    # print(dracor_ids[138])
    # print(vector_names)
    # print(vector_names[1120])


    # print(type(matrix))
    # print(matrix)



    # #  cluster plotting:
    # #elb.elbow_plot(data=matrix, no_of_clusters=5)
    # sil.silhouette_plot(data=matrix, no_of_clusters=5)
    cluster_scatterplot(top_termns=25)





