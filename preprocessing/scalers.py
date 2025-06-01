"""Scalers for preprocessing data."""

import polars as pl

class StandardScaler:
  """Standardizes features by removing the mean and scaling to unit variance.

  Attributes:
    mean_: Polars Series, a Series of mean values for each column in the fitted
      data.
    scale_: Polars Series, a Series of standard deviation values for each column
      in the fitted data.
  """

  def __init__(self):
    """Initializes StandardScaler with mean_ and scale_ set to None."""
    self.mean_ = None
    self.scale_ = None

  def fit(self, X: pl.DataFrame):
    """Computes the mean and standard deviation to be used for later scaling.

    Args:
      X: Polars DataFrame, the data used to compute the mean and standard
        deviation.

    Returns:
      self: Fitted scaler.
    """
    self.mean_ = X.mean()
    self.scale_ = X.std()
    # Handle potential division by zero if a column has zero standard deviation
    self.scale_ = self.scale_.map_elements(lambda x: 1.0 if x == 0 else x)
    return self

  def transform(self, X: pl.DataFrame) -> pl.DataFrame:
    """Performs standardization by centering and scaling.

    Args:
      X: Polars DataFrame, the data to standardize.

    Returns:
      Polars DataFrame, the transformed data.
    """
    if self.mean_ is None or self.scale_ is None:
      raise RuntimeError("Scaler has not been fitted yet.")
    return (X - self.mean_) / self.scale_

  def fit_transform(self, X: pl.DataFrame) -> pl.DataFrame:
    """Fits to data, then transforms it.

    Args:
      X: Polars DataFrame, the data to fit and transform.

    Returns:
      Polars DataFrame, the transformed data.
    """
    return self.fit(X).transform(X)


class MinMaxScaler:
  """Transforms features by scaling each feature to a given range.

  This estimator scales and translates each feature individually such
  that it is in the given range on the training set, e.g., between zero and one.

  Attributes:
    min_: Polars Series, per feature adjustment for minimum. Calculated as
      `feature_range[0] - data_min_ * self.scale_`.
    scale_: Polars Series, per feature relative scaling of the data. Calculated as
      `(feature_range[1] - feature_range[0]) / (data_max_ - data_min_)`.
    feature_range: tuple (min, max), default=(0, 1)
      Desired range of transformed data.
  """

  def __init__(self, feature_range=(0, 1)):
    """Initializes MinMaxScaler with feature_range, min_ and scale_ set to None.

    Args:
      feature_range: tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    """
    self.min_ = None
    self.scale_ = None
    self.feature_range = feature_range

  def fit(self, X: pl.DataFrame):
    """Computes the minimum and maximum to be used for later scaling.

    Args:
      X: Polars DataFrame, the data used to compute the per-feature minimum and
        maximum.

    Returns:
      self: Fitted scaler.
    """
    data_min_ = X.min()
    data_max_ = X.max()
    data_range_ = data_max_ - data_min_

    # Handle potential division by zero if a column has zero range
    self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range_.map_elements(
        lambda x: 1.0 if x == 0 else x
    )
    # Ensure that if data_range_ was 0, scale_ is 1 to avoid issues,
    # effectively making it a no-op for that feature if feature_range is (0,1)
    # or shifting if feature_range is different.
    # A more robust way might be to check data_range_ directly.
    for i, val in enumerate(data_range_.to_list()):
        if val == 0:
            self.scale_ = self.scale_.with_row_count().with_columns(
                pl.when(pl.col("row_nr") == i)
                .then(pl.lit(1.0, dtype=self.scale_.dtype)) # Set scale to 1 if range is 0
                .otherwise(pl.col(self.scale_.name))
                .alias(self.scale_.name)
            )[self.scale_.name]


    self.min_ = self.feature_range[0] - data_min_ * self.scale_
    return self

  def transform(self, X: pl.DataFrame) -> pl.DataFrame:
    """Scales features of X according to feature_range.

    Args:
      X: Polars DataFrame, input data that will be transformed.

    Returns:
      Polars DataFrame, transformed data.
    """
    if self.min_ is None or self.scale_ is None:
      raise RuntimeError("Scaler has not been fitted yet.")
    return X * self.scale_ + self.min_

  def fit_transform(self, X: pl.DataFrame) -> pl.DataFrame:
    """Fits to data, then transforms it.

    Args:
      X: Polars DataFrame, the data to fit and transform.

    Returns:
      Polars DataFrame, the transformed data.
    """
    return self.fit(X).transform(X)
