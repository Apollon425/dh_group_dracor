from sklearn.cluster import KMeans # clustering algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# evaluating k
# elbow plot: inertia = sum of squared distances of samples to their closest cluster center; decreases with number of clusters
# ideally: low inertia, as few clusters as possible
# 



def elbow_plot(data, no_of_clusters,  plotsize=(10,10)):

    cluster_range = list(range(2, no_of_clusters+1))
    print(cluster_range)
    inertia_list = []
    for n in cluster_range:
        k_means = KMeans(n_clusters=n, random_state=42)
        k_means.fit(data)
        inertia_list.append(k_means.inertia_)
        
    # plotting
    plot = plt.figure(figsize=plotsize)
    ax = plot.add_subplot(111)
    sns.lineplot(y=inertia_list, x=cluster_range, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Inertia")
    ax.set_xticks(cluster_range)
    plot.show()
    plot.savefig("elbow_plot_dracor.png")



