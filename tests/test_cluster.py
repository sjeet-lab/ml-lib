"""Tests for clustering algorithms."""

import numpy as np  # Add numpy import
import pytest

from custom_ml_library.cluster import DBSCAN


class TestDBSCAN:
    """Tests for the DBSCAN clustering algorithm."""

    @pytest.fixture
    def sample_cluster_data(self):
        """Simple dataset for clustering."""
        # Example from DBSCAN docstring
        X_list = [(3, 3), (20, 20), (21, 25), (-5, 1), (1, 1), (2, 2), (10, 1)]
        # Convert to Polars DataFrame for consistency, though DBSCAN currently takes list of tuples
        # DBSCAN should ideally be updated to accept Polars DataFrame
        # For now, we'll test with the list of tuples it expects.
        # X = pl.DataFrame({"x": [p[0] for p in X_list], "y": [p[1] for p in X_list]})
        return X_list

    def test_dbscan_fit(self, sample_cluster_data):
        """Test DBSCAN fit method and basic output."""
        X = sample_cluster_data
        eps = 6
        min_points = 3
        dbs = DBSCAN(eps=eps, min_points=min_points)

        # The current DBSCAN.fit takes X and returns (n_clusters, labels)
        # And also sets self.labels and self.n_clusters
        n_clusters_returned, labels_returned = dbs.fit(X)

        expected_labels = [0, -1, -1, 0, 0, 0, -1]  # From docstring
        expected_n_clusters = 1  # From docstring

        assert dbs.n_clusters_ == expected_n_clusters
        np.testing.assert_array_equal(dbs.labels_, expected_labels)
        assert n_clusters_returned == expected_n_clusters
        np.testing.assert_array_equal(labels_returned, expected_labels)

    def test_dbscan_str_representation(self):
        """Test the __str__ method of DBSCAN."""
        dbs = DBSCAN(eps=0.5, min_points=5)
        assert str(dbs) == "eps = 0.5, min_points = 5"

    # Future tests could include:
    # - Testing with Polars DataFrame input (after refactoring DBSCAN)
    # - Edge cases (all noise, all one cluster, multiple clusters)
    # - Data with different scales
