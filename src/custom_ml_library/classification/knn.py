"""K-Nearest Neighbors Classifier."""

from collections import Counter
import numpy as np
import polars as pl

from ..utils.distances import euclidean_distance


class KNeighborsClassifier:
    """Classifier implementing the k-nearest neighbors vote.

    Attributes:
      n_neighbors: int, number of neighbors to use by default for :meth:`predict`.
      X_train_: np.ndarray, feature data used during :meth:`fit`.
      y_train_: np.ndarray, target values used during :meth:`fit`.
    """

    def __init__(self, n_neighbors: int = 5):
        """Initializes KNeighborsClassifier.

        Args:
          n_neighbors: Number of neighbors to use by default for :meth:`predict`.
        """
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than 0.")
        self.n_neighbors = n_neighbors
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """Fits the k-nearest neighbors classifier from the training dataset.

        Args:
          X: Polars DataFrame of shape (n_samples, n_features), training data.
          y: Polars Series of shape (n_samples,), target values.

        Returns:
          self: Fitted estimator.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]})"
                " do not match."
            )
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"Expected n_neighbors <= n_samples, but got n_neighbors ="
                f" {self.n_neighbors} and n_samples = {X.shape[0]}"
            )

        self.X_train_ = X.to_numpy()
        self.y_train_ = y.to_numpy()
        return self

    def _predict_single(self, x_test_point: np.ndarray) -> any:
        """Predicts the class label for a single test point.

        Args:
          x_test_point: A NumPy array representing a single data point to classify.

        Returns:
          The predicted class label for the input data point.
        """
        distances = [
            euclidean_distance(x_test_point, x_train_point)
            for x_train_point in self.X_train_
        ]
        # Get indices of the k nearest neighbors
        k_neighbor_indices = np.argsort(distances)[: self.n_neighbors]
        # Get labels of these neighbors
        k_neighbor_labels = [self.y_train_[i] for i in k_neighbor_indices]
        # Determine the most frequent label (majority vote)
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """Predicts the class labels for the provided data.

        Args:
          X: Polars DataFrame of shape (n_samples, n_features), test samples.

        Returns:
          Polars Series of shape (n_samples,), predicted class labels for each data
          sample.

        Raises:
          RuntimeError: If the model has not been fitted yet.
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError(
                "This KNeighborsClassifier instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

        X_np = X.to_numpy()
        if X_np.ndim == 1:  # Single sample prediction
            X_np = X_np.reshape(1, -1)

        predictions = [self._predict_single(x_test_point) for x_test_point in X_np]
        return pl.Series(values=predictions, name="predictions")
