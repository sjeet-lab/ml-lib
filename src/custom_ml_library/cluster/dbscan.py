"""Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm."""

# Standard library imports
# (none in this file)

# Third-party imports
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Local application/library specific imports
from ..utils.distances import euclidean_distance

# The r"""...""" block below is a diagram and not the module docstring.
r"""
  ____        _   _____   _____   _____
 / ___|      | | | ____| | ____| |_   _|
 \___ \   _  | | |  _|   |  _|     | |
  ___) | | |_| | | |___  | |___    | |
 |____/   \___/  |_____| |_____|   |_|

"""


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_points : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
    n_clusters_ : int
        The number of clusters found by the algorithm. If labels are all -1,
        this is 0.

    Examples
    --------
    >>> X = [(3,3), (20,20), (21,25), (-5,1), (1,1), (2,2),(10,1)]
    >>> dbs = DBSCAN(eps=6, min_points=3)
    >>> dbs.fit(X)
    (1, [0, -1, -1, 0, 0, 0, -1])
    >>> dbs.n_clusters_
    1
    >>> dbs.labels_
    array([ 0, -1, -1,  0,  0,  0, -1])
    """

    def __init__(self, eps: float = 0.5, min_points: int = 5):
        """Initialize DBSCAN."""
        self.eps = eps
        self.min_points = min_points
        self.labels_ = np.array([])  # Initialize as empty numpy array
        self.n_clusters_ = 0

    def fit(self, X: list[tuple[float, ...]]):
        """Perform DBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : list of tuples or list of lists
            A list of data points. Each point is a tuple or list of numbers.
            Example: [(x1, y1), (x2, y2), ...]

        Returns
        -------
        n_clusters : int
            Number of clusters found.
        labels : np.ndarray
            Cluster labels for each point. Noisy samples are given the label -1.
        """
        # Using self.labels_ and self.n_clusters_ to store results, consistent with scikit-learn
        current_labels = []  # Use a temporary list for labels during fitting

        # Convert points to NumPy arrays for compatibility with custom euclidean_distance
        # and ensure consistent data type for calculations.
        X_np = [np.asarray(point, dtype=float) for point in X]

        dist_matrix_rows = []
        for point_i in X_np:
            dist_row = [
                1 if euclidean_distance(point_i, point_j) <= self.eps else 0
                for point_j in X_np
            ]
            dist_matrix_rows.append(dist_row)

        if not dist_matrix_rows:  # Handle empty input X
            self.labels_ = np.array([])
            self.n_clusters_ = 0
            return self.n_clusters_, self.labels_

        graph = csr_matrix(dist_matrix_rows)
        # Get connected components.
        # n_components is the number of connected components
        # component_labels is an array of labels for each point.
        _, component_labels = connected_components(
            csgraph=graph, directed=False, return_labels=True
        )

        # Count occurrences of each component label
        unique_labels, counts = np.unique(component_labels, return_counts=True)
        labels_count = dict(zip(unique_labels, counts))

        for label_val in component_labels:
            if labels_count[label_val] < self.min_points:
                current_labels.append(-1)  # Noise point
            else:
                current_labels.append(label_val)  # Core or border point

        self.labels_ = np.asarray(current_labels)

        # Count unique cluster labels (excluding noise -1)
        unique_cluster_labels = {label for label in self.labels_ if label != -1}
        self.n_clusters_ = len(unique_cluster_labels)

        # Optional: Re-map cluster labels to be 0-indexed and contiguous.
        if self.n_clusters_ > 0:
            sorted_unique_clusters = sorted(list(unique_cluster_labels))
            label_mapping = {
                old_label: new_label
                for new_label, old_label in enumerate(sorted_unique_clusters)
            }
            mapped_labels = [
                label_mapping[label] if label != -1 else -1 for label in self.labels_
            ]
            self.labels_ = np.asarray(mapped_labels)

        return self.n_clusters_, self.labels_

    def __str__(self) -> str:
        """Return string representation of DBSCAN parameters."""
        return f"eps = {self.eps}, min_points = {self.min_points}"
