"""Utility functions for calculating distances."""

import numpy as np
import polars as pl

def euclidean_distance(point1, point2) -> float:
  """Calculates the Euclidean distance between two points.

  The points can be Polars Series or 1-D NumPy arrays of the same length.

  Args:
    point1: A Polars Series or 1-D NumPy array representing the first point.
    point2: A Polars Series or 1-D NumPy array representing the second point.

  Returns:
    A float representing the Euclidean distance between the two points.

  Raises:
    ValueError: If the input points have different lengths or are not 1-D.
  """
  if isinstance(point1, pl.Series):
    point1 = point1.to_numpy()
  if isinstance(point2, pl.Series):
    point2 = point2.to_numpy()

  if not isinstance(point1, np.ndarray) or not isinstance(point2, np.ndarray):
    raise TypeError("Inputs must be Polars Series or NumPy ndarrays.")

  if point1.ndim != 1 or point2.ndim != 1:
    raise ValueError("Input arrays must be 1-dimensional.")

  if len(point1) != len(point2):
    raise ValueError("Input points must have the same length.")

  return np.sqrt(np.sum((point1 - point2)**2))
