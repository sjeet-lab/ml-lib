"""Tests for utility functions."""

import numpy as np
import polars as pl
import pytest # For raising errors

from utils.distances import euclidean_distance


class TestEuclideanDistance:
  """Tests for the euclidean_distance function."""

  def test_with_polars_series(self):
    """Tests euclidean_distance with Polars Series as input."""
    point1 = pl.Series("p1", [1, 2, 3])
    point2 = pl.Series("p2", [4, 5, 6])
    # Expected: sqrt((1-4)^2 + (2-5)^2 + (3-6)^2)
    #         = sqrt((-3)^2 + (-3)^2 + (-3)^2)
    #         = sqrt(9 + 9 + 9) = sqrt(27) approx 5.196
    expected_distance = np.sqrt(27)
    np.testing.assert_almost_equal(
        euclidean_distance(point1, point2), expected_distance, decimal=6
    )

  def test_with_numpy_arrays(self):
    """Tests euclidean_distance with NumPy arrays as input."""
    point1 = np.array([1, 2, 3])
    point2 = np.array([4, 5, 6])
    expected_distance = np.sqrt(27)
    np.testing.assert_almost_equal(
        euclidean_distance(point1, point2), expected_distance, decimal=6
    )

  def test_with_known_values(self):
    """Tests euclidean_distance with known values (Pythagorean triple)."""
    point1 = pl.Series("p1", [0, 0])
    point2 = pl.Series("p2", [3, 4])
    # Expected: sqrt((0-3)^2 + (0-4)^2) = sqrt(9 + 16) = sqrt(25) = 5
    expected_distance = 5.0
    np.testing.assert_almost_equal(
        euclidean_distance(point1, point2), expected_distance, decimal=6
    )

    point1_np = np.array([0,0])
    point2_np = np.array([3,4])
    np.testing.assert_almost_equal(
        euclidean_distance(point1_np, point2_np), expected_distance, decimal=6
    )


  def test_mixed_inputs_series_and_array(self):
    """Tests euclidean_distance with a mix of Polars Series and NumPy array."""
    point1_pl = pl.Series("p1", [1, 2, 3])
    point2_np = np.array([4, 5, 6])
    expected_distance = np.sqrt(27)
    np.testing.assert_almost_equal(
        euclidean_distance(point1_pl, point2_np), expected_distance, decimal=6
    )

    point1_np = np.array([1, 2, 3])
    point2_pl = pl.Series("p2", [4, 5, 6])
    np.testing.assert_almost_equal(
        euclidean_distance(point1_np, point2_pl), expected_distance, decimal=6
    )


  def test_raises_error_for_different_lengths(self):
    """Tests that euclidean_distance raises ValueError for inputs of different lengths."""
    point1 = pl.Series("p1", [1, 2, 3])
    point2 = pl.Series("p2", [4, 5])
    with pytest.raises(ValueError, match="Input points must have the same length."):
      euclidean_distance(point1, point2)

    point1_np = np.array([1,2,3])
    point2_np = np.array([4,5])
    with pytest.raises(ValueError, match="Input points must have the same length."):
      euclidean_distance(point1_np, point2_np)

  def test_raises_error_for_non_1d_inputs(self):
    """Tests that euclidean_distance raises ValueError for non 1-D inputs."""
    point1 = np.array([[1, 2], [3, 4]])
    point2 = np.array([1,2])
    with pytest.raises(ValueError, match="Input arrays must be 1-dimensional."):
        euclidean_distance(point1, point2)

    point1_pl = pl.Series([pl.Series([1,2]), pl.Series([3,4])]) # This is not how you make 2D polars for this
    # For Polars, the to_numpy conversion would typically handle Series of Series,
    # but the internal check is on ndim.
    # A Polars DataFrame would be more appropriate for 2D data, but the function
    # is designed for Series. If a Series contains list-like elements, to_numpy()
    # might create an object array which then fails ndim check or later math.
    # The current implementation expects Series of scalars.

    # Test with a Series that when converted to numpy is 2D
    # This is tricky with polars series, as they are strictly 1D.
    # The check is primarily for numpy arrays.

  def test_raises_error_for_invalid_types(self):
    """Tests that euclidean_distance raises TypeError for invalid input types."""
    point1 = [1, 2, 3] # Python list
    point2 = np.array([4, 5, 6])
    with pytest.raises(TypeError, match="Inputs must be Polars Series or NumPy ndarrays."):
      euclidean_distance(point1, point2)

    point1_pl = pl.Series([1,2,3])
    point2_str = "not a point"
    with pytest.raises(TypeError, match="Inputs must be Polars Series or NumPy ndarrays."):
      euclidean_distance(point1_pl, point2_str)

  def test_zero_length_input(self):
    """Tests euclidean_distance with zero-length inputs."""
    point1 = pl.Series(dtype=pl.Float64)
    point2 = pl.Series(dtype=pl.Float64)
    expected_distance = 0.0
    np.testing.assert_almost_equal(euclidean_distance(point1, point2), expected_distance)

    point1_np = np.array([])
    point2_np = np.array([])
    np.testing.assert_almost_equal(euclidean_distance(point1_np, point2_np), expected_distance)
