"""Module for sample normalization."""

from typing import Optional, List

import numpy as np
import polars as pl


class Normalizer:
    """Normalize samples individually to unit norm.

    Each sample (i.e. each row of the data matrix) with at least one
    non-zero component is rescaled independently of other samples so that
    its norm (l1, l2) equals one.

    This transformer is able to work with dense Polars DataFrames.
    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance.

    Args:
        norm (str, 'l1' or 'l2', default='l2'):
            The norm to use to normalize each non-zero sample.

    Attributes:
        n_features_in_ (int):
            Number of features seen during `fit` (or first `transform` call).
        feature_names_in_ (List[str]):
            Names of features seen during `fit` (or first `transform` call).
    """

    def __init__(self, norm: str = 'l2'):
        if norm not in ('l1', 'l2'):
            raise ValueError("Norm must be 'l1' or 'l2'")
        self.norm = norm
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None):
        """Only validates the data and stores feature info. Does not learn anything.

        Args:
            X (pl.DataFrame): The data to validate. Shape (n_samples, n_features).
            y (pl.DataFrame, optional): Ignored. Present for API consistency.

        Returns:
            self: Fitted normalizer.
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(X)}")

        # Store feature names and count even for empty DataFrame to define schema
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        if X.height == 0: # No data to validate dtypes from
            return self

        for col_name in X.columns:
            if not X[col_name].dtype.is_numeric():
                raise TypeError(
                    f"Column '{col_name}' has non-numeric type "
                    f"{X[col_name].dtype}. Normalizer only supports numeric data."
                )
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Scale each non-zero sample to unit norm.

        Args:
            X (pl.DataFrame): The data to normalize. Shape (n_samples, n_features).

        Returns:
            pl.DataFrame: The transformed data.

        Raises:
            TypeError: If X is not a Polars DataFrame or contains non-numeric data.
            ValueError: If X has a different number of features than fit.
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(X)}")

        if self.n_features_in_ is None: # Fit hasn't been called explicitly
            # This behavior (implicit fit on transform) is common but can sometimes be surprising.
            # For Normalizer, it's acceptable as fit doesn't learn from data values.
            self.fit(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but Normalizer "
                f"was expecting {self.n_features_in_} features based on previous data."
            )
        # Ensure column order and names match what was seen in fit
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                "Feature names or order of X do not match those seen during fit. "
                f"Expected: {self.feature_names_in_}, Got: {X.columns}"
            )

        if X.height == 0:
            return X.clone() # Return empty DataFrame with same schema

        # Re-check dtypes in transform as well, in case fit was on an empty DF
        # and this is the first non-empty data seen.
        for col_name in X.columns:
            if not X[col_name].dtype.is_numeric():
                raise TypeError(
                    f"Column '{col_name}' has non-numeric type "
                    f"{X[col_name].dtype}. Normalizer only supports numeric data."
                )

        X_np = X.to_numpy()

        if self.norm == 'l2':
            norms = np.linalg.norm(X_np, axis=1, keepdims=True)
        elif self.norm == 'l1':
            # For l1 norm, ensure we are summing absolute values
            norms = np.sum(np.abs(X_np), axis=1, keepdims=True)
        else:
            # This case should ideally be unreachable due to __init__ validation
            raise ValueError(f"Invalid norm '{self.norm}' encountered in transform.")

        # Avoid division by zero for samples with zero norm (all features are zero)
        # These samples will remain zero.
        norms[norms == 0.0] = 1.0

        X_normalized_np = X_np / norms

        return pl.DataFrame(X_normalized_np, schema=self.feature_names_in_, orient="row")

    def fit_transform(self, X: pl.DataFrame, y: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Fit to data, then transform it.

        Args:
            X (pl.DataFrame): The data to fit and transform.
            y (pl.DataFrame, optional): Ignored. Present for API consistency.

        Returns:
            pl.DataFrame: Transformed data.
        """
        return self.fit(X, y).transform(X)
