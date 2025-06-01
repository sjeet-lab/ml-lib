r"""
  ____        _   _____   _____   _____
 / ___|      | | | ____| | ____| |_   _|
 \___ \   _  | | |  _|   |  _|     | |
  ___) | | |_| | | |___  | |___    | |
 |____/   \___/  |_____| |_____|   |_|

"""

import numpy as np  # Added numpy import
from ..utils.distances import euclidean_distance

# from scipy.spatial import distance # Removed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class DBSCAN:
    """
    Example:
    [In]: X = [(3,3), (20,20), (21,25), (-5,1), (1,1), (2,2),(10,1) ]
          eps = 6
          min_points = 3
          dbs = DBSCAN(6,3)
          dbs.fit(X)
    [Out]:(1, [0, -1, -1, 0, 0, 0, -1])

    [In]: dbs.n_clusters
    [Out]:1
    [In]: dbs.labels
    [Out]:[0, -1, -1, 0, 0, 0, -1]
    """

    def __init__(self, eps=0.1, min_points=3):
        self.eps = eps
        self.min_points = min_points

    def fit(self, X):
        dist = []
        self.labels = []
        self.n_clusters = 0
        # counter = [] # Removed as it's unused after refactoring n_clusters logic

        # Convert points to NumPy arrays for compatibility with custom euclidean_distance
        X_np = [np.asarray(point, dtype=float) for point in X]

        for point_i in X_np:
            d_row = []  # Changed variable name from d to d_row for clarity
            for point_j in X_np:
                if euclidean_distance(point_i, point_j) <= self.eps:
                    d_row.append(1)
                else:
                    d_row.append(0)
            dist.append(d_row)

        graph = csr_matrix(dist)
        # Get connected components.
        # init_comps[0] is the number of connected components
        # init_comps[1] is an array of labels for each point.
        _, lbl = connected_components(csgraph=graph, directed=False, return_labels=True)

        labels_count = {
            val: list(lbl).count(val) for val in np.unique(lbl)
        }  # Use np.unique for efficiency

        for val_lbl in lbl:
            if labels_count[val_lbl] < self.min_points:
                self.labels.append(-1)  # Noise point
            else:
                self.labels.append(
                    val_lbl
                )  # Core or border point, assign component label

        # Count unique cluster labels (excluding noise -1)
        unique_cluster_labels = {label for label in self.labels if label != -1}
        self.n_clusters = len(unique_cluster_labels)

        # Optional: Re-map cluster labels to be 0-indexed if they are not already
        # For now, using the direct connected components labels.

        return (self.n_clusters, self.labels)

    def __str__(self):
        return "eps = {}, min_points = {}".format(self.eps, self.min_points)
