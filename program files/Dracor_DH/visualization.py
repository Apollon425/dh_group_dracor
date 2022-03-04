from configparser import MissingSectionHeaderError
from turtle import shape
import dracor_data as dr
import pandas as pd
from matplotlib import pyplot as plt
import sys
import elbow as elb
import silhouette as sil




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

if __name__ == '__main__':

    #  metadata plotting:
    #draw_plot(data=read_data_csv(METADATA_PATH), column="averageClustering", plot_type="scatter", annotate=False, first_year=1850)

    #  calculate tif-idf:
    return_list = dr.get_features("ita",  vocab=True, syntax=False, lemmatize=True, get_ids= True)
    matrix = return_list[0]
    dracor_ids = return_list[1]
    vector_names = return_list[2]


    #  cluster plotting:
    elb.elbow_plot(data=matrix, no_of_clusters=5)
    #sil.silhouette_plot(matrix=matrix, no_of_clusters=5)





