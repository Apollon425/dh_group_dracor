import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
import dracor_data as dr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys


vectors, dracor_ids, vector_names = dr.get_features("ger", get_ids= True, remove_stopwords=True)


k = 10
model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1)

model.fit(vectors)

#order_centroids = model.cluster_centers_.argsort()[:, ::-1]


kmean_indices = model.fit_predict(vectors)
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())


colors = ["r", "b", "c", "y", "m", "g", "k", "yellow", "greenyellow", "pink"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fix, ax = plt.subplots(figsize=(50, 50))

ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

# for i, txt in enumerate(names):
#     ax.annotate(txt[0:5], (x_axis[i], y_axis[i]))

plt.savefig("ger_no_stopwords.png")

# test
