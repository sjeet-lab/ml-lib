"""Module for encoding categorical features and labels."""

from typing import Optional, List, Any

import numpy as np
import polars as pl


class LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1.

    This transformer should be used to encode target values, *y*, and not the input *X*.

    Attributes:
        classes_ (np.ndarray): Holds the label for each class.
    """

    def __init__(self):
        self.classes_: Optional[np.ndarray] = None

    def fit(self, y: pl.Series):
        """Fit label encoder.

        Args:
            y (pl.Series): Target values.

        Returns:
            self: Fitted encoder.
        """
        if not isinstance(y, pl.Series):
            raise TypeError(f"Expected Polars Series for y, got {type(y)}")
        if y.is_empty():
            raise ValueError("Cannot fit LabelEncoder with an empty Series.")

        self.classes_ = y.unique().sort().to_numpy()
        return self

    def transform(self, y: pl.Series) -> pl.Series:
        """Transform labels to normalized encoding.

        Args:
            y (pl.Series): Target values.

        Returns:
            pl.Series: Transformed labels.

        Raises:
            RuntimeError: If the encoder has not been fitted yet.
            ValueError: If y contains labels not seen during fit.
        """
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder has not been fitted yet.")
        if not isinstance(y, pl.Series):
            raise TypeError(f"Expected Polars Series for y, got {type(y)}")

        class_to_int = {cls_val: i for i, cls_val in enumerate(self.classes_)}

        try:
            # Ensure all values in y are hashable for dictionary lookup
            # This check might be more robust depending on expected Series dtypes
            transformed_values = [class_to_int[val] for val in y]
        except KeyError as e:
            unseen_label = e.args[0]
            raise ValueError(f"y contains previously unseen label: {unseen_label}") from e
        except TypeError as e: # Handles unhashable types in y
            raise TypeError(f"Values in y must be hashable to be used as dictionary keys. Got error: {e}") from e


        return pl.Series(values=transformed_values, name=y.name, dtype=pl.Int64)

    def fit_transform(self, y: pl.Series) -> pl.Series:
        """Fit label encoder and return encoded labels.

        Args:
            y (pl.Series): Target values.

        Returns:
            pl.Series: Transformed labels.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y: pl.Series) -> pl.Series:
        """Transform labels back to original encoding.

        Args:
            y (pl.Series): Transformed labels (integer encoded).

        Returns:
            pl.Series: Original labels.

        Raises:
            RuntimeError: If the encoder has not been fitted yet.
            ValueError: If y contains labels outside the range of fitted classes.
        """
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder has not been fitted yet.")
        if not isinstance(y, pl.Series):
            raise TypeError(f"Expected Polars Series for y, got {type(y)}")
        if not y.dtype.is_integer(): # Checks for any integer type (Int8, Int16, Int32, Int64, UInt variants)
            raise ValueError(f"Input y for inverse_transform must be integer type, got {y.dtype}")

        min_val, max_val = 0, len(self.classes_) - 1

        if y.is_empty(): # Handle empty series before min/max
            # Determine original dtype for the output Series
            # Create a temporary Polars Series from self.classes_ to infer its dtype
            # This assumes self.classes_ is not None, which is checked above.
            original_dtype = pl.Series(values=self.classes_).dtype # Infer dtype from classes_
            return pl.Series(values=[], name=y.name, dtype=original_dtype)

        # Check for out-of-bounds values if series is not empty
        # TODO: Polars min/max on empty series will return None. Add check for that or ensure y is not empty before.
        # The y.is_empty() check above handles this.
        if y.min() < min_val or y.max() > max_val:
            # Filter for problematic values to show in the error message
            problem_values = y.filter((pl.col(y.name) < min_val) | (pl.col(y.name) > max_val))
            raise ValueError(
                f"y contains values outside of the range of encoded classes. "
                f"Problematic values: {problem_values.to_list()}"
            )

        original_values = self.classes_[y.to_numpy()] # Indexing numpy array with polars Series directly might be slow
                                                      # y.to_numpy() is correct here.
        original_dtype = pl.Series(values=self.classes_).dtype # Infer dtype from classes_

        return pl.Series(values=original_values, name=y.name, dtype=original_dtype)


class OneHotEncoder:
    """Encode categorical features as a one-hot numeric array.

    The input to this transformer should be a Polars DataFrame containing
    integer-like features, where the integers represent categories.
    The features are encoded using a one-hot (or dummy) encoding scheme.
    This creates a binary column for each category and returns a DataFrame
    with these new columns.

    Args:
        handle_unknown (str, 'error' or 'ignore', default='error'):
            Whether to raise an error or ignore if an unknown categorical feature
            is present during transform.
        drop (None, 'first', or 'if_binary', default=None):
            Specifies a methodology to use to drop one category per feature.
            - None: Retain all features.
            - 'first': Drop the first category in each feature. If a feature
              has only one category, it will be dropped.
            - 'if_binary': Drop the first category in each feature with two
              categories. Features with 1 or more than 2 categories are
              unaffected.
            Note: `drop` is not fully implemented in this version beyond placeholder.
                  Currently behaves as if `drop=None`.

    Attributes:
        categories_ (List[np.ndarray]):
            The categories mapped to each feature, in order.
            Each element in the list is a NumPy array of unique categories
            for the corresponding feature.
        n_features_in_ (int):
            Number of features seen during `fit`.
        feature_names_in_ (List[str]):
            Names of features seen during `fit`.
        _feature_names_out (List[str]):
            Internal storage for output feature names.
    """

    def __init__(self, *, handle_unknown: str = 'error', drop: Optional[str] = None):
        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown must be 'error' or 'ignore'")
        # Basic support for drop, full implementation can be complex
        if drop is not None and drop not in ['first', 'if_binary']: # TODO: Implement drop logic
            raise ValueError("drop must be None, 'first', or 'if_binary'")

        self.handle_unknown = handle_unknown
        self.drop = drop # Note: Advanced drop logic is not implemented yet.
        self.categories_: Optional[List[np.ndarray]] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self._feature_names_out: Optional[List[str]] = None

    def fit(self, X: pl.DataFrame, y: Optional[Any] = None):
        """Fit OneHotEncoder to X.

        Args:
            X (pl.DataFrame): The data to determine the categories of each feature.
                              Each column should contain integer categories.
            y (Any, optional): Ignored. Present for API consistency.

        Returns:
            self: Fitted encoder.
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(X)}")
        if X.is_empty():
            # It's important to define what fitting on an empty DataFrame means.
            # Should it store n_features_in_ as X.width and empty categories?
            # Or raise error? Scikit-learn would typically store n_features_in_
            # and feature_names_in_ but categories_ would be problematic.
            # For now, let's be strict and require non-empty for meaningful category learning.
            # However, the prompt implies it could be fitted on empty.
            # Let's allow fitting on empty DF, storing schema but no categories.
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = X.columns
            self.categories_ = [np.array([]) for _ in X.columns] # Empty categories for each column
            self._generate_output_feature_names(X.columns) # Will generate no names if categories are empty
            return self


        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns
        self.categories_ = []

        for col_name in X.columns:
            column = X.select(col_name).to_series()
            # We expect categories to be discrete. Unique + sort captures them.
            # No specific dtype check, relying on Polars' ability to handle unique().sort().
            cats = column.unique().sort().drop_nulls().to_numpy() # drop_nulls for categories
            self.categories_.append(cats)

        self._generate_output_feature_names(X.columns)
        return self

    def _generate_output_feature_names(self, input_features: List[str]):
        """Generates output feature names based on categories."""
        self._feature_names_out = []
        if self.categories_ is None:
            return

        for i, cats_for_feature in enumerate(self.categories_):
            input_col_name = input_features[i]
            # TODO: Implement self.drop logic here when generating names
            # For now, drop=None behavior
            for cat_val in cats_for_feature:
                self._feature_names_out.append(f"{input_col_name}_{cat_val}")

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names for transformation.

        Args:
            input_features (List[str], optional): Unused, kept for compatibility.
                                                 Uses feature_names_in_ from fit.

        Returns:
            List[str]: Output feature names.
        """
        if self._feature_names_out is None: # Should be generated by fit
             # This state indicates fit() might not have completed properly or was on empty data
             # with no columns, leading to _feature_names_out not being set.
             if self.feature_names_in_ is not None: # If we know input names but not output
                self._generate_output_feature_names(self.feature_names_in_)

        # If still None (e.g. fit on DF with no columns), return empty list
        return self._feature_names_out if self._feature_names_out is not None else []


    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform X using one-hot encoding.

        Args:
            X (pl.DataFrame): The data to encode.

        Returns:
            pl.DataFrame: Transformed data with one-hot encoded columns.

        Raises:
            RuntimeError: If the encoder has not been fitted yet.
            ValueError: If X has different features than during fit or unknown categories.
        """
        if self.categories_ is None or self.feature_names_in_ is None or \
           self.n_features_in_ is None or self._feature_names_out is None:
            raise RuntimeError("OneHotEncoder has not been fitted yet. Call fit first.")

        if not isinstance(X, pl.DataFrame):
            raise TypeError(f"Expected Polars DataFrame, got {type(X)}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneHotEncoder "
                f"was fitted with {self.n_features_in_} features."
            )
        if list(X.columns) != list(self.feature_names_in_): # Check column order and names
            raise ValueError(
                "Feature names or order of X do not match those seen during fit. "
                f"Expected: {self.feature_names_in_}, Got: {X.columns}"
            )

        if X.is_empty():
            # Return empty DataFrame with correct one-hot encoded column names and types
            return pl.DataFrame(schema={name: pl.UInt8 for name in self._feature_names_out})


        one_hot_encoded_series_list = []

        for i, col_name in enumerate(X.columns):
            column_series = X.select(col_name).to_series()
            fitted_categories = self.categories_[i]

            # Handle unknown categories encountered in the current column_series
            # This check should happen before attempting to create columns for each category
            if self.handle_unknown == 'error':
                current_column_categories_set = set(column_series.unique().drop_nulls().to_list())
                fitted_categories_set = set(fitted_categories)
                unknowns = current_column_categories_set - fitted_categories_set
                if unknowns:
                    raise ValueError(
                        f"Found unknown categories {unknowns} in column '{col_name}' during transform."
                    )

            # TODO: Implement self.drop logic here when creating series
            # For now, drop=None behavior
            for category_val in fitted_categories:
                new_col_name = f"{col_name}_{category_val}"
                # Create a new Series for this category: 1 if value matches, 0 otherwise
                # Using pl.col().eq().cast() is more idiomatic Polars than map_elements for this
                one_hot_series = X.select(
                    pl.col(col_name).eq(pl.lit(category_val)).cast(pl.UInt8).alias(new_col_name)
                ).to_series()
                one_hot_encoded_series_list.append(one_hot_series)

        if not one_hot_encoded_series_list:
             # This could happen if input X had columns but all categories_ lists were empty
             # (e.g. fit on a DataFrame with columns but all values were nulls and drop_nulls() was used)
            return pl.DataFrame(schema={name: pl.UInt8 for name in self._feature_names_out}) # Return with correct schema

        return pl.concat(one_hot_encoded_series_list, how='horizontal')

    def fit_transform(self, X: pl.DataFrame, y: Optional[Any] = None) -> pl.DataFrame:
        """Fit OneHotEncoder to X, then transform X.

        Args:
            X (pl.DataFrame): Input data.
            y (Any, optional): Ignored.

        Returns:
            pl.DataFrame: Transformed data.
        """
        return self.fit(X, y).transform(X)
