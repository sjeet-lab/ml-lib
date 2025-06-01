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
