"""Module for basic imputation strategies."""

from typing import Optional, List, Any, Union

import numpy as np # For np.nan and potentially other np functions if needed
import polars as pl


class SimpleImputer:
    """Impute missing values.

    Imputation transformer for completing missing values using common strategies.
    Missing values can be represented by `None`, `np.nan`, or `float('nan')`.

    Args:
        missing_values (Any, default=None):
            The placeholder for the missing values. All occurrences of
            `missing_values` will be imputed. For Polars, `None` is typically
            used to represent nulls which Polars handles natively.
        strategy (str, default='mean'):
            The imputation strategy.
            - If "mean", then replace missing values using the mean along
              each column. Can only be used with numeric data.
            - If "median", then replace missing values using the median along
              each column. Can only be used with numeric data.
            - If "most_frequent", then replace missing using the most frequent
              value along each column. Can be used with string or numeric data.
            - If "constant", then replace missing values with `fill_value`.
              Can be used with string or numeric data.
        fill_value (Union[str, int, float], optional):
            When strategy == "constant", `fill_value` is used to replace all
            occurrences of missing_values. If left to the default, `fill_value`
            will be 0 when imputing numeric data and "missing_value" for
            string data.
        add_indicator (bool, default=False):
            If True, a `MissingIndicator` transform will stack onto output of
            the imputer's transform. This allows a predictive estimator to
            account for missingness despite imputation. (Not implemented in this version)

    Attributes:
        statistics_ (pl.DataFrame, optional):
            The imputation fill value for each feature. This is a 1-row DataFrame.
            Computed when strategy is "mean", "median", or "most_frequent".
        fill_value_ (Any):
            The actual fill value used when strategy is "constant".
        n_features_in_ (int):
            Number of features seen during `fit`.
        feature_names_in_ (List[str]):
            Names of features seen during `fit`.
    """

    def __init__(self,
                 missing_values: Any = None, # Polars handles None/null natively
                 strategy: str = 'mean',
                 fill_value: Optional[Union[str, int, float]] = None,
                 add_indicator: bool = False): # add_indicator not implemented

        if strategy not in ['mean', 'median', 'most_frequent', 'constant']:
            raise ValueError(
                "Strategy must be one of 'mean', 'median', "
                "'most_frequent', or 'constant'."
            )
        if strategy == 'constant' and fill_value is None:
            # Default fill_value behavior will be determined in fit based on dtype
            pass
        if add_indicator:
            raise NotImplementedError("add_indicator=True is not yet implemented.")

        self.missing_values = missing_values # Note: Polars handles its own nulls. This arg might be less relevant.
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator

        self.statistics_: Optional[pl.DataFrame] = None
        self.fill_value_: Any = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: pl.DataFrame, y: Optional[Any] = None):
        """Fit the imputer on X.

        Args:
            X (pl.DataFrame): Input data, where `missing_values` are imputed.
            y (Any, optional): Ignored.

        Returns:
            self: Fitted imputer.
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(X)}")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        if X.is_empty(): # No data to compute statistics or determine fill_value type
            if self.strategy == "constant" and self.fill_value is not None:
                self.fill_value_ = self.fill_value
            # For other strategies, statistics_ will remain None or empty if we can't infer types.
            # This might require handling in transform if fitted on empty and then transforming non-empty.
            # For now, let's assume if fit on empty, transform on empty is the main valid path.
            # If transform is called on non-empty, it might need to re-fit or error if no stats.
            self.statistics_ = pl.DataFrame({name: [] for name in X.columns}) # Empty DF with schema
            return self

        if self.strategy in ['mean', 'median']:
            for col_name in X.columns:
                if not X[col_name].dtype.is_numeric():
                    raise ValueError(
                        f"Strategy '{self.strategy}' can only be used with numeric data. "
                        f"Column '{col_name}' has type {X[col_name].dtype}."
                    )
            if self.strategy == 'mean':
                self.statistics_ = X.mean()
            else: # median
                self.statistics_ = X.median()
        elif self.strategy == 'most_frequent':
            stats_list = []
            for col_name in X.columns:
                mode_series = X[col_name].mode()
                # mode() can return multiple values if ties; pick the first.
                # mode() can return empty if all nulls; handle this.
                if mode_series.is_empty():
                    # If all nulls, fill strategy needs a default.
                    # This depends on dtype or a predefined fill_value for this case.
                    # For now, let's fill with a default based on inferred type or a placeholder.
                    # This is tricky: if all null, what's the mode?
                    # Pylint might complain if `val` is not defined.
                    val = None # Fallback if mode is empty
                    if X[col_name].dtype.is_numeric(): val = 0
                    elif X[col_name].dtype == pl.Utf8: val = "missing"
                    # This could be improved by allowing user to specify this fallback
                    stats_list.append(pl.Series(col_name, [val], dtype=X[col_name].dtype))

                else:
                    stats_list.append(pl.Series(col_name, [mode_series[0]], dtype=X[col_name].dtype))
            if stats_list:
                 self.statistics_ = pl.concat(stats_list, how="horizontal")
            else: # Should not happen if X is not empty and has columns
                 self.statistics_ = pl.DataFrame({name: [] for name in X.columns})


        elif self.strategy == 'constant':
            # If fill_value is provided, use it. Otherwise, 0 for numeric, "missing_value" for string.
            # This needs to be determined per column if fill_value is None.
            # For simplicity if fill_value is given, it's stored directly.
            # If not, we'd need to create a statistics_ like DF.
            if self.fill_value is not None:
                self.fill_value_ = self.fill_value # User-provided fill_value
                # We don't need per-column statistics_ if fill_value_ is a single scalar to be broadcasted.
                # However, to be consistent for transform, statistics_ could hold this value for each column.
                self.statistics_ = pl.DataFrame(
                    [pl.Series(c, [self.fill_value_], dtype=X[c].dtype.base_type()) for c in X.columns]
                )

            else: # Determine default fill_value per column
                default_fills = []
                for col_name in X.columns:
                    col_dtype = X[col_name].dtype
                    if col_dtype.is_numeric():
                        default_fills.append(pl.Series(col_name, [0], dtype=col_dtype).cast(X[col_name].dtype.base_type()))
                    else: # Assume Utf8 or other non-numeric as string-like
                        default_fills.append(pl.Series(col_name, ["missing_value"], dtype=pl.Utf8).cast(X[col_name].dtype.base_type()))
                if default_fills:
                    self.statistics_ = pl.concat(default_fills, how="horizontal")
                else: # Should not happen if X is not empty and has columns
                    self.statistics_ = pl.DataFrame({name: [] for name in X.columns})


        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Impute all missing values in X.

        Args:
            X (pl.DataFrame): The input data with missing values.

        Returns:
            pl.DataFrame: DataFrame with imputed values.
        """
        if self.n_features_in_ is None or self.feature_names_in_ is None:
            # This implies fit was not called or was called on an empty DF without columns
            # If fit was on empty DF with columns, n_features_in_ and feature_names_in_ are set.
            # A more robust check might be on statistics_ or fill_value_
            if self.strategy == 'constant' and self.fill_value_ is not None:
                 # If strategy is constant and fill_value_ is set, we might proceed
                 # assuming X schema matches implicit expectation or allow transform
                 # to define schema if this is the first data seen.
                 # However, for consistency with other scalers, let's be strict.
                 pass # Allow proceeding if constant fill_value_ is set
            else:
                 raise RuntimeError(
                    "SimpleImputer has not been fitted with valid statistics yet. "
                    "Call fit or ensure fit was on non-empty data."
                )


        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(X)}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SimpleImputer "
                f"was fitted with {self.n_features_in_} features."
            )
        if list(X.columns) != list(self.feature_names_in_):
             raise ValueError(
                "Feature names or order of X do not match those seen during fit. "
                f"Expected: {self.feature_names_in_}, Got: {X.columns}"
            )
        if X.is_empty():
            return X.clone()


        fill_map = {}
        if self.strategy == 'constant':
            if self.fill_value_ is not None: # User provided single fill_value
                for col_name in X.columns:
                    fill_map[col_name] = self.fill_value_
            else: # Default fill_values were stored in statistics_
                if self.statistics_ is None or self.statistics_.is_empty():
                     raise RuntimeError("Imputer not fitted or fitted on empty data without default fill values determined.")
                for col_name in self.statistics_.columns:
                    fill_map[col_name] = self.statistics_.item(0, col_name)
        else: # mean, median, most_frequent
            if self.statistics_ is None or self.statistics_.is_empty():
                raise RuntimeError("Imputer not fitted or statistics are not computed (e.g. fit on empty data).")
            for col_name in self.statistics_.columns:
                 fill_map[col_name] = self.statistics_.item(0, col_name)

        # Polars fill_null uses the provided value directly.
        # The `missing_values` parameter is less relevant here as Polars uses its own null representation.
        # We are filling Polars' native nulls.

        # Create a list of expressions for fill_null
        fill_exprs = [
            pl.col(c_name).fill_null(fill_map[c_name]).alias(c_name)
            for c_name in X.columns
        ]
        return X.with_columns(fill_exprs)


    def fit_transform(self, X: pl.DataFrame, y: Optional[Any] = None) -> pl.DataFrame:
        """Fit to data, then transform it.

        Args:
            X (pl.DataFrame): Input data to impute.
            y (Any, optional): Ignored.

        Returns:
            pl.DataFrame: DataFrame with imputed values.
        """
        return self.fit(X, y).transform(X)
