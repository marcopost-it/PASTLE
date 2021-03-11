from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import numpy as np

class Random():
    def __init__(self, X, n_pivots):
        index = np.random.choice(X.shape[0], n_pivots, replace=False)
        self.pivots = X[index]


class Perturbation():
    def __init__(self, x, n_pivots, perc_range):
        self.pivots = np.empty((n_pivots, len(x)))
        for f in range(len(x)):
            for p in range(n_pivots):
                self.pivots[p, f] = np.random.uniform(x[f] - x[f] * perc_range, x[f] + x[f] * perc_range)


class Clustering():
    def __init__(self, X, n_clusters):
        normalized_X = preprocessing.normalize(X)
        self.n_clusters = n_clusters
        clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.cluster_assignments = clusterer.fit_predict(normalized_X)
        pivots = []
        for idx in range(self.n_clusters):
            pivots.append(np.mean(X[self.cluster_assignments == idx, :], axis=0))
        self.pivots = np.array(pivots)