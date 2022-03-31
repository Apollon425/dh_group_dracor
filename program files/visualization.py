from configparser import MissingSectionHeaderError
from turtle import shape
from xmlrpc.client import Boolean

import sklearn
import dracor_data as dr
import pandas as pd
from matplotlib import pyplot as plt 

import matplotlib.cm as cm
import os
from pathlib import Path

import sys
import dracor_nlp

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
import string
import dracor_data as dr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import seaborn as sns



class Visualization:

    OUTPUT_PATH_BASE = 'visualization_output/clustering/'
    out_path = ""



    def __init__(self, corpus: str, text: str, min_df: int, remove_Stopwords: bool, lemmatize: bool,

                top_centroids: int, label: str, clusters: int, feature_domain: str): 

                self.corpus = corpus
                self.text = text
                self.min_df = min_df
                self.remove_Stopwords = remove_Stopwords
                self.lemmatize = lemmatize
                self.feature_domain = feature_domain

                self.top_centroids = top_centroids
                self.label = label
                self.clusters = clusters
                self.out_path = self.define_output_path()


    def create_output_folder(self):
        """Folder for all outputs of visualizations."""

        path_exists = os.path.exists(self.out_path)

        if not path_exists:    
            os.makedirs(Path(self.out_path))
            print("New output folder created.")
        else:
            print("No new output folder created, since one with the same name already exists.")


    def define_output_path(self):
        """Defines outputpath based on object variable settings to store variations of data sets separately."""

        stopword_str = "noStop" if self.remove_Stopwords is True else ""
        out_string = self.OUTPUT_PATH_BASE + f"{self.feature_domain}" + f'/{self.corpus}/{self.text}_{stopword_str}_min_df={str(self.min_df)}_cluster={str(self.clusters)}'
        return out_string


    def set_time_frame(self, first_year, last_year, corpus) -> tuple:

        """help function for metadata_plot, determines time frame to draw plot for; if not specified by user, it defaults to the oldest/youngest play in the corpus 
        """
        path = dr.GER_METADATA_PATH if corpus == "ger" else dr.ITA_METADATA_PATH

        if not first_year:
            first_year = dr.read_data_csv(path)['yearNormalized'].min()
        if not last_year:
            last_year = dr.read_data_csv(path)['yearNormalized'].max()


        return first_year, last_year


    def metadata_plot(self, data: pd.DataFrame, column: str, plot_type: str, annotate: bool, first_year: int = None, last_year: int = None, corpus="ger") -> None:

        """
        :param column: y-axis value to plot against time; must be column name out of given data frame (see parameter data')
        :param plot_type: name of plot-type; currently only supports scatter-plot
        :param annotate: if True, data points are labeled (currently with name of the author)
        """

        first_year, last_year = self.set_time_frame(first_year, last_year, corpus)
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


    def write_centroids(self, clusters, order_centroids, top_centroids, df: pd.DataFrame) -> None:
        """Prints centroids for each cluster, saves them as csv-file."""

        centroid_outer = []  
        centroid_headers = []
      

        for i in range(clusters):
            centroid_list = []
            centroid_headers.append(f"Cluster {i}")

            for centroid_index in order_centroids[i][:top_centroids]:

                centroid_list.append(df.columns[centroid_index])

            centroid_outer.append(centroid_list)
            
        centroid_df = pd.DataFrame(centroid_outer)
        centroid_df = centroid_df.transpose()
        centroid_df.columns = centroid_headers
        print("Centroids: ")
        print(centroid_df)
        file_path = Path(self.out_path + f"/Top_{top_centroids}_centroids.csv")
        dr.write_to_csv(path=file_path, data=centroid_df, encoding='utf-8', index=False, header=centroid_headers)


    def construct_df(self) -> pd.DataFrame:
        """Uses dracor_data.py to make API calls to dracor.org and read the corpus data from there.
           Returns the full dataframe of all feature vectors combined, as well as those features separately for each feature domain (tf-idf, pos, metadata)
           Returns dracor ids of the plays and the vector names of the tf-idf vectors as well.
        """
            

        pos, matrix, dracor_ids, vector_names, meta_features = dr.get_features(corpus = self.corpus,
                                                                                text = self.text,
                                                                                remove_stopwords = self.remove_Stopwords, 
                                                                                min_df= self.min_df, 
                                                                                lemmatize = self.lemmatize)


        #  build dataframe for each feauture domain (tf-idf, pos, meta-data and a dataframe with all possible features):

        tf_idf_df = dr.convert_scipymatrix_to_dataframe(matrix, vector_names)
        meta_df = meta_features
        df = pd.concat([tf_idf_df, meta_features], axis=1)
        pos_df = dr.dict_to_df(pos)
        df = pd.concat([df, pos_df], axis=1)
        df = df.drop(['id'], axis=1)
        print("final full df:")
        print(df)
        return df, tf_idf_df, pos_df, dracor_ids, vector_names, meta_df


    def print_cluster_content(self, df: pd.DataFrame, meta: pd.DataFrame) -> None:
        """Prints the plays belonging to the clusters, saves that classification as csv-file."""
    
        for cluster in range(self.clusters):
            cluster_content = df.query(f'kmean_indices=={cluster}')['dracor_ids'].to_list()
            meta_data_cluster = meta.loc[meta['id'].isin(cluster_content)]
            print(f"\n Metadata for Cluster {cluster}: \n\n {meta_data_cluster} \n ------------------- \n")
            meta_data_cluster.to_csv(self.out_path + f"/cluster {cluster}.csv")    


    def get_metadata(self) -> pd.DataFrame:
        if self.corpus == "ger":
            meta = dr.read_data_csv(dr.GER_METADATA_PATH)
        elif self.corpus == "ita":
            meta = dr.read_data_csv(dr.ITA_METADATA_PATH)
        else:
            sys.exit("Corpus name invalid. Only \"ger\" and \"ita\" are supported.")
        return meta


    def cluster_scatterplot(self, df: pd.DataFrame, dracor_ids) -> None:
        """Draws scatterplot of dataframe."""


        #  1) cluster data using k-means:

        model = KMeans(n_clusters=self.clusters, init="k-means++", n_init=1, random_state=10).fit(df)  #  max_iter = 100
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]  #  sort centroids for each cluster

        try:
            self.write_centroids(clusters=self.clusters, order_centroids=order_centroids, top_centroids=self.top_centroids, df=df)
        except (FileNotFoundError, PermissionError, IndexError) as e:
            print(e)
            print("Error writing centroids.")

        print("df before cluster scatter")

        #  2) dimension reduction feature vectors:

        pca = PCA(n_components=2)
        scatter_plot_points = pca.fit_transform(df)

        #  3) build new dataframe for more convenient plotting:

        x_axis = [o[0] for o in scatter_plot_points]
        y_axis = [o[1] for o in scatter_plot_points]
        kmean_indices = model.fit_predict(df)
        plot_df = pd.DataFrame(list(zip(x_axis, y_axis, kmean_indices, dracor_ids)), columns=["x_axis", "y_axis", "kmean_indices", "dracor_ids"])

        print("Plot this dataframe:")
        print(plot_df)


        #  4) plot that df:

        metadata = self.get_metadata()

        plot = sns.relplot(data = plot_df, x = 'x_axis', y = 'y_axis', hue = 'kmean_indices', palette = 'tab10', kind = 'scatter', height=15, aspect=1.5)
        if self.label is not None:
            ax = plot.axes[0, 0]
            for idx, row in plot_df.iterrows():  #  iterate over rows of df, get id and then look up a label to mark the data point in the plot with it
                x = row['x_axis']
                y = row['y_axis']
                dracor_id_of_this_row = row['dracor_ids']
                label_name = metadata.loc[metadata['id'] == f"{dracor_id_of_this_row}", f'{self.label}'].item()  #  get label (for ex. 'firstAuthor' depending on value of self.label)
                label_name = label_name + ", " + str((metadata.loc[metadata['id'] == f"{dracor_id_of_this_row}", f'yearNormalized'].item()))  #  add yearNormalized to label name (firstAuthor, year)
                ax.text(x+25, y-10, label_name, horizontalalignment='left')



        #  5) save it:
        label_str = "_lab" if self.label is not None else ""
        plt.savefig(Path(self.out_path + f"/cluster_plot{label_str}.png"))

        #  6) find contents of clusters, save them as csv

        self.print_cluster_content(df=plot_df, meta=metadata)


    def silhouette_plot(self, data: pd.DataFrame) -> None:
       # this code is based on the sci-kit learn tutorial for silhouette plot: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    
        """Outputs as many silhouette plot's of data in dataframe as clusters are specified in 'clusters'-attribute in the visualization object. 
           Used to evaluate the quality of the clustering. Outputs silhouette scores as well."""


        range_n_clusters = list(range(2, self.clusters+2))
        for n_clusters in range_n_clusters:
            #  Draw Silhouette plot and cluster-scatterplot for each cluster in range
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot (silhouette coefficient can range from -1 to 1)
            ax1.set_xlim([-0.1, 1])           
            ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])  #  The (n_clusters+1)*10 is for inserting blank space between silhouette

            clusterer = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, random_state=10)
            cluster_labels = clusterer.fit_predict(data)  
            silhouette_avg = silhouette_score(data, cluster_labels)  
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            
            sample_silhouette_values = silhouette_samples(data, cluster_labels)  #  Compute the silhouette scores for each sample  

            #  Aggregate the silhouette scores for samples belonging to clusters and sort them:
            y_lower = 10
            for i in range(n_clusters):

                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  #  Label the silhouette plots with their cluster numbers


                #  Compute the new y_lower for next plot
                y_lower = y_upper + 10  #  10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            #  The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            #  2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

            pca = PCA(n_components=2)
            scatter_plot_points = pca.fit_transform(data)
            x_axis = [o[0] for o in scatter_plot_points]
            y_axis = [o[1] for o in scatter_plot_points]

            ax2.scatter(
                x_axis, y_axis, marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
                                                                                            
            )

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

        plt.show()


    def elbow_plot(self, data: pd.DataFrame, plotsize=(10,10)):
        """Outputs an elbow plot of data in dataframe. Used to determine how many clusters to use."""
        
        # The tutorial by The Programming Historian was useful for the Elbow Plot algorithm

        data = data.drop(data.filter(['dracor_id']), axis=1)

        cluster_range = list(range(2, self.clusters+2))
        inertia_list = []
        for n in cluster_range:
            k_means = KMeans(n_clusters=n, random_state=42)
            k_means.fit(data)
            inertia_list.append(k_means.inertia_)
            
        #  plotting
        plot = plt.figure(figsize=plotsize)
        ax = plot.add_subplot(111)
        sns.lineplot(y=inertia_list, x=cluster_range, ax=ax)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Inertia")
        ax.set_xticks(cluster_range)
        plot.savefig(Path(self.out_path + "/elbow_plot_dracor.png"))  


if __name__ == '__main__':

    visualizer = Visualization(

                                corpus = "ita",
                                text = "spoken",
                                min_df = 20,
                                remove_Stopwords = True,
                                lemmatize = True,
                                top_centroids = 10,
                                label = 'firstAuthor',  #  set to None if no label on the datapoints in the cluster plot is desired
                                clusters = 21,
                                feature_domain = "tf-idf"   #  "all_features" or "pos" or "tf-idf" or "meta"
    )

    #  1)  get data, construct df:

    df_all_features, tf_idf_df, pos_df, dracor_ids, vector_names, meta_df = visualizer.construct_df()

    #  2)  select features to include:

    if visualizer.feature_domain == "pos":
        df=pos_df
    elif visualizer.feature_domain == "tf-idf":
        df=tf_idf_df
    elif visualizer.feature_domain == "meta":
        df=meta_df
    elif visualizer.feature_domain == "all_features":
        df=df_all_features
    else:
        sys.exit("Invalid feature domain.")


    #  3)  visualize it and save it:

    visualizer.create_output_folder()
    #visualizer.cluster_scatterplot(df=df, dracor_ids=dracor_ids) 
    #visualizer.elbow_plot(data=df)
    visualizer.silhouette_plot(data=df)

