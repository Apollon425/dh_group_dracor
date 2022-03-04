from sklearn.cluster import KMeans # clustering algorithms
import matplotlib.pyplot as plt
import seaborn as sns

# evaluating k
# elbow plot: inertia = sum of squared distances of samples to their closest cluster center; decreases with number of clusters
# ideally: low inertia, as few clusters as possible
# 

def elbow_plot(range, data, plotsize=(10,10)):

    inertia_list = []
    for n in range:
        k_means = KMeans(n_clusters=n, random_state=42)
        k_means.fit(data)
        inertia_list.append(k_means.inertia_)
        
    # plotting
    plot = plt.figure(figsize=plotsize)
    ax = plot.add_subplot(111)
    sns.lineplot(y=inertia_list, x=range, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(range))
    plot.show()
    plot.savefig("elbow_plot_dracor.png")
