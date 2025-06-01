"""Feature scaling transformers.

This module includes common feature scalers:
- StandardScaler: Standardizes features by removing the mean and scaling to unit variance.
- MinMaxScaler: Scales features to a given range (e.g., 0-1).
- RobustScaler: Scales features using statistics robust to outliers (median and IQR).
"""

import polars as pl
import numpy as np # Added for RobustScaler docstring example
from typing import Tuple, List, Optional # Added for RobustScaler type hints


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
        self.mean_ = X.mean()  # This is a 1-row DataFrame
        std_dev = X.std()  # This is also a 1-row DataFrame
        # Handle potential division by zero if a column has zero standard deviation
        # Iterate through columns and apply the condition expression
        self.scale_ = std_dev.select(
            [
                pl.when(pl.col(col_name) == 0)
                .then(1.0)
                .otherwise(pl.col(col_name))
                .alias(col_name)
                for col_name in std_dev.columns
            ]
        )
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

        # Apply transformation column by column using expressions
        # to ensure broadcasting or scalar operations are clear.
        transformed_cols_exprs = []
        for col_name in X.columns:
            # self.mean_ and self.scale_ are 1-row DataFrames.
            # We need to extract the scalar value for each column for the operation.
            mean_val = self.mean_.item(0, col_name)
            scale_val = self.scale_.item(0, col_name)

            transformed_cols_exprs.append(
                ((pl.col(col_name) - mean_val) / scale_val).alias(col_name)
            )
        return X.select(transformed_cols_exprs)

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
        data_min_ = X.min()  # DataFrame: shape (1, n_features)
        data_max_ = X.max()  # DataFrame: shape (1, n_features)
        data_range_ = data_max_ - data_min_  # DataFrame: shape (1, n_features)

        # Calculate scale for each column
        scale_expressions = []
        for col_name in data_range_.columns:
            # Get the actual value from the 1-row DataFrame for data_range_
            # Assuming data_range_ has only one row.
            # If X was empty, data_range_ might be empty or have different shape.
            # For robust handling, one might check data_range_.is_empty() or shape.
            # However, standard behavior is X won't be empty for fit.
            range_val = data_range_[
                0, col_name
            ]  # Accesses the single value in the column

            if range_val == 0:
                scale_expressions.append(pl.lit(1.0).alias(col_name))
            else:
                scale_value = (
                    self.feature_range[1] - self.feature_range[0]
                ) / range_val
                scale_expressions.append(pl.lit(scale_value).alias(col_name))

        if not data_range_.columns:  # Handle empty input DataFrame X
            self.scale_ = pl.DataFrame()
            self.min_ = pl.DataFrame()
        else:
            self.scale_ = data_range_.select(
                scale_expressions
            )  # self.scale_ is a 1-row DataFrame
            # self.min_ should also be a 1-row DataFrame
            # self.feature_range[0] is a scalar. data_min_ and self.scale_ are DataFrames.
            # Correct operation: scalar - (DataFrame * DataFrame)
            # self.scale_ and data_min_ are 1-row DataFrames.
            product = data_min_ * self.scale_
            # self.min_ should be a 1-row DataFrame as well.
            self.min_ = product.select(
                [
                    (pl.lit(self.feature_range[0]) - pl.col(col_name)).alias(col_name)
                    for col_name in product.columns
                ]
            )

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

        # Apply transformation column by column using expressions
        transformed_cols_exprs = []
        for col_name in X.columns:
            # self.scale_ and self.min_ are 1-row DataFrames.
            # Extract scalar value for each column for the operation.
            scale_val = self.scale_.item(0, col_name)
            min_val = self.min_.item(0, col_name)

            transformed_cols_exprs.append(
                (pl.col(col_name) * scale_val + min_val).alias(col_name)
            )
        return X.select(transformed_cols_exprs)

    def fit_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Fits to data, then transforms it.

        Args:
          X: Polars DataFrame, the data to fit and transform.

        Returns:
          Polars DataFrame, the transformed data.
        """
        return self.fit(X).transform(X)


class RobustScaler:
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to the
    quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile) and
    the 3rd quartile (75th quantile).

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Median and
    quantile range are then stored to be used on later data using the
    `transform` method.

    Args:
        with_centering (bool, default=True):
            If True, center the data before scaling.
        with_scaling (bool, default=True):
            If True, scale the data to interquartile range.
        quantile_range (Tuple[float, float], default=(25.0, 75.0)):
            Quantile range used to calculate `scale_`.
            Must be between 0 and 100.

    Attributes:
        center_ (pl.DataFrame, optional):
            The median value for each feature in the training set.
            Stored when `with_centering` is True.
        scale_ (pl.DataFrame, optional):
            The (scaled) interquartile range for each feature in the training set.
            Stored when `with_scaling` is True.
        n_features_in_ (int):
            Number of features seen during `fit`.
        feature_names_in_ (List[str]):
            Names of features seen during `fit`.
    """

    def __init__(self, *,
                 with_centering: bool = True,
                 with_scaling: bool = True,
                 quantile_range: Tuple[float, float] = (25.0, 75.0)):
        if not (0.0 <= quantile_range[0] < quantile_range[1] <= 100.0):
            raise ValueError(
                "Invalid quantile_range: "
                f"{quantile_range}. Values must be between 0 and 100, "
                "and q_min < q_max."
            )
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_: Optional[pl.DataFrame] = None
        self.scale_: Optional[pl.DataFrame] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None):
        """Compute the median and quantile range to be used for later scaling.

        Args:
            X (pl.DataFrame): The data used to compute the median and
                interquartile range. Shape (n_samples, n_features).
            y (pl.DataFrame, optional): Ignored. Present for API consistency.

        Returns:
            self: Fitted scaler.
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(X)}")
        if X.is_empty():
            # Handle empty DataFrame input: set attributes to indicate no fitting.
            self.n_features_in_ = 0
            self.feature_names_in_ = []
            self.center_ = pl.DataFrame() if self.with_centering else None
            self.scale_ = pl.DataFrame() if self.with_scaling else None
            return self


        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        if self.with_centering:
            self.center_ = X.median() # Polars DataFrame.median() gives a 1-row DF

        if self.with_scaling:
            q_min, q_max = self.quantile_range
            # Polars quantile expects value between 0.0 and 1.0
            quantiles_min = X.quantile(q_min / 100.0)
            quantiles_max = X.quantile(q_max / 100.0)

            self.scale_ = quantiles_max - quantiles_min

            # Handle cases where scale is zero (e.g., constant feature)
            # Create a boolean mask for zero scale values
            # This needs to be done carefully if self.scale_ could be empty
            if self.scale_.is_empty(): # Should not happen if X was not empty
                 pass # Or set to an empty DF with correct schema if possible
            else:
                zero_scale_mask_exprs = []
                for c_name in self.scale_.columns:
                    # Ensure we're comparing with appropriate dtype if necessary
                    # For numeric types, direct comparison with 0 should be fine.
                    zero_scale_mask_exprs.append((pl.col(c_name) == 0).alias(c_name))

                if zero_scale_mask_exprs: # If there are columns to process
                    zero_scale_mask = self.scale_.select(zero_scale_mask_exprs)

                    # Update scale_: where mask is True, set to 1.0, else keep original
                    update_scale_exprs = []
                    for c_name in self.scale_.columns:
                        original_col_dtype = self.scale_[c_name].dtype
                        update_scale_exprs.append(
                            pl.when(zero_scale_mask[c_name])
                            .then(pl.lit(1.0, dtype=original_col_dtype))
                            .otherwise(self.scale_[c_name])
                            .alias(c_name)
                        )
                    if update_scale_exprs:
                        self.scale_ = self.scale_.select(update_scale_exprs)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Center and scale the data.

        Args:
            X (pl.DataFrame): The data to transform. Shape (n_samples, n_features).

        Returns:
            pl.DataFrame: The transformed data.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
            ValueError: If the number of features in X is different from fit or names differ.
        """
        if self.n_features_in_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit first.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"was fitted with {self.n_features_in_} features."
            )
        # Ensure column order and names match what was seen in fit
        # This is a stricter check than just set equality of names.
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                "Feature names or order of X do not match those seen during fit. "
                f"Expected: {self.feature_names_in_}, Got: {X.columns}"
            )

        X_transformed = X.clone() # Avoid modifying original DataFrame

        if self.with_centering:
            if self.center_ is None: # Should not happen if fitted and with_centering
                raise RuntimeError("Scaler not fitted or not fitted with centering.")
            if self.center_.is_empty() and self.n_features_in_ > 0 : # Fitted on empty data with features
                 raise RuntimeError("Scaler fitted on empty data; center is undefined for transformation.")
            if not self.center_.is_empty():
                X_transformed = X_transformed.select([
                    (pl.col(c_name) - self.center_.item(0, c_name)).alias(c_name)
                    for c_name in X_transformed.columns
                ])


        if self.with_scaling:
            if self.scale_ is None: # Should not happen if fitted and with_scaling
                raise RuntimeError("Scaler not fitted or not fitted with scaling.")
            if self.scale_.is_empty() and self.n_features_in_ > 0:
                raise RuntimeError("Scaler fitted on empty data; scale is undefined for transformation.")
            if not self.scale_.is_empty():
                X_transformed = X_transformed.select([
                    (pl.col(c_name) / self.scale_.item(0, c_name)).alias(c_name)
                    for c_name in X_transformed.columns
                ])

        return X_transformed

    def fit_transform(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Fit to data, then transform it.

        Args:
            X (pl.DataFrame): The data to fit and transform.
            y (pl.DataFrame, optional): Ignored. Present for API consistency.

        Returns:
            pl.DataFrame: Transformed data.
        """
        return self.fit(X, y).transform(X)
