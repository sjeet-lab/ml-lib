"""Tests for classification models."""

import numpy as np
import polars as pl
import pytest

from custom_ml_library.classification import KNeighborsClassifier

# Assuming utils.distances is importable, and __init__.py files are set up
# from custom_ml_library.utils import euclidean_distance # This import is not directly used in this test file.


class TestKNeighborsClassifier:
    """Tests for the KNeighborsClassifier class."""

    @pytest.fixture
    def sample_data(self):
        """Simple dataset with two distinct clusters."""
        X = pl.DataFrame(
            {
                "feature1": [1.0, 1.5, 2.0, 5.0, 5.5, 6.0, 1.2, 5.2],
                "feature2": [1.0, 1.5, 1.0, 4.0, 4.5, 4.0, 0.8, 4.2],
            }
        )
        y = pl.Series("labels", [0, 0, 0, 1, 1, 1, 0, 1])
        return X, y

    def test_fit_and_predict(self, sample_data):
        """Tests fit and predict methods."""
        X, y = sample_data
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        # Test that X_train_ and y_train_ are stored correctly
        np.testing.assert_array_equal(model.X_train_, X.to_numpy())
        np.testing.assert_array_equal(model.y_train_, y.to_numpy())

        # Predict on training data (should have high accuracy)
        # For k=3, and this simple data, it should be perfect.
        predictions_train = model.predict(X)
        assert isinstance(predictions_train, pl.Series)
        assert predictions_train.name == "predictions"
        np.testing.assert_array_equal(predictions_train.to_numpy(), y.to_numpy())

        # Predict on new, known points
        # Point near cluster 0
        new_point_0 = pl.DataFrame({"feature1": [1.1], "feature2": [1.1]})
        prediction_0 = model.predict(new_point_0)
        assert prediction_0.item() == 0

        # Point near cluster 1
        new_point_1 = pl.DataFrame({"feature1": [5.1], "feature2": [4.1]})
        prediction_1 = model.predict(new_point_1)
        assert prediction_1.item() == 1

        # Predict multiple new points
        new_points = pl.DataFrame(
            {"feature1": [1.1, 5.1, 0.9, 6.1], "feature2": [1.1, 4.1, 1.2, 3.8]}
        )
        expected_new_predictions = pl.Series([0, 1, 0, 1])
        predictions_new = model.predict(new_points)
        np.testing.assert_array_equal(
            predictions_new.to_numpy(), expected_new_predictions.to_numpy()
        )

    def test_different_n_neighbors(self, sample_data):
        """Tests with different values of n_neighbors."""
        X, y = sample_data

        # k=1, should still classify correctly for this separable data
        model_k1 = KNeighborsClassifier(n_neighbors=1)
        model_k1.fit(X, y)
        predictions_k1 = model_k1.predict(X)
        np.testing.assert_array_equal(predictions_k1.to_numpy(), y.to_numpy())

        # k=5, might change some boundary points if any were ambiguous
        # For X[6] = (1.2, 0.8) label 0. Distances to others:
        # (1,1)->0: 0.28
        # (1.5,1.5)->0: 0.76
        # (2,1)->0: 0.82
        # (5,4)->1: 4.95
        # (5.5,4.5)->1: 5.6
        # (6,4)->1: 5.8
        # (1.2,0.8)->0: 0  (self)
        # (5.2,4.2)->1: 5.25
        # Neighbors for X[6] (1.2, 0.8) with k=3 (excluding self if not careful, but predict handles new data)
        # If we predict on X itself:
        # For point (1.2, 0.8), label 0.
        # Distances: [0.28 (0), 0.76 (0), 0.82 (0), 4.95 (1), 5.6 (1), 5.8 (1), 0 (0), 5.25 (1)]
        # Sorted indices (by distance): self, (1,1), (1.5,1.5), (2,1), (5,4), (5.2,4.2), (5.5,4.5), (6,4)
        # Labels:                     0,    0,       0,        0,      1,       1,        1,        1
        # k=3 neighbors (excluding self): (1,1), (1.5,1.5), (2,1) -> all 0. Pred: 0. Correct.
        # k=5 neighbors (excluding self): (1,1), (1.5,1.5), (2,1), (5,4), (5.2,4.2) -> 0,0,0,1,1. Pred: 0. Correct.

        model_k5 = KNeighborsClassifier(n_neighbors=5)
        model_k5.fit(X, y)
        predictions_k5 = model_k5.predict(X)
        np.testing.assert_array_equal(predictions_k5.to_numpy(), y.to_numpy())

    def test_predict_before_fit_raises_error(self, sample_data):
        """Tests that predict before fit raises a RuntimeError."""
        X, _ = sample_data
        model = KNeighborsClassifier()
        with pytest.raises(
            RuntimeError, match="This KNeighborsClassifier instance is not fitted yet."
        ):
            model.predict(X)

    def test_invalid_n_neighbors(self):
        """Tests that KNeighborsClassifier raises ValueError for invalid n_neighbors."""
        with pytest.raises(ValueError, match="n_neighbors must be greater than 0."):
            KNeighborsClassifier(n_neighbors=0)
        with pytest.raises(ValueError, match="n_neighbors must be greater than 0."):
            KNeighborsClassifier(n_neighbors=-1)

    def test_n_neighbors_greater_than_samples(self, sample_data):
        """Tests error when n_neighbors is greater than number of samples in fit."""
        X, y = sample_data  # X has 8 samples
        model = KNeighborsClassifier(n_neighbors=10)
        with pytest.raises(ValueError, match="Expected n_neighbors <= n_samples"):
            model.fit(X, y)

    def test_mismatched_X_y_lengths_in_fit(self):
        """Tests error when X and y have different number of samples in fit."""
        X = pl.DataFrame({"f1": [1, 2, 3]})
        y = pl.Series([0, 1])
        model = KNeighborsClassifier()
        with pytest.raises(
            ValueError, match="Number of samples in X .* and y .* do not match."
        ):
            model.fit(X, y)

    def test_single_sample_prediction(self, sample_data):
        """Tests predicting a single sample passed as a DataFrame."""
        X, y = sample_data
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        single_X = X.head(1)  # DataFrame with one row
        prediction = model.predict(single_X)

        assert isinstance(prediction, pl.Series)
        assert len(prediction) == 1
        assert prediction.item() == y[0]

    def test_predict_output_type_and_name(self, sample_data):
        """Tests predict output type and series name."""
        X, y = sample_data
        model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        predictions = model.predict(X)
        assert isinstance(predictions, pl.Series)
        assert predictions.name == "predictions"

    def test_fit_returns_self(self, sample_data):
        """Tests that fit returns self."""
        X, y = sample_data
        model = KNeighborsClassifier()
        assert model.fit(X, y) is model


# Example of how euclidean_distance might be used if it wasn't directly imported
# but part of the module structure. For now, direct import is fine.
# from ..utils import distances # if tests was a sub-package of the main lib
# Or: from project_name.utils import distances
# For this structure: from utils.distances import euclidean_distance
# No need for specific tests for euclidean_distance here as they are in test_utils.py
# but KNN relies on it.
