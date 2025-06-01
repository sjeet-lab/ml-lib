"""Tests for imputation strategies."""

import numpy as np
import polars as pl
import pytest

from custom_ml_library.preprocessing import SimpleImputer


class TestSimpleImputer:
    """Tests for the SimpleImputer class."""

    @pytest.fixture
    def numeric_data_with_nulls(self):
        """Numeric Polars DataFrame with nulls."""
        return pl.DataFrame({
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [10.0, None, 30.0, None, 50.0],
            "c": [None, None, None, None, None] # All nulls
        }, schema={"a": pl.Float64, "b": pl.Float64, "c": pl.Float64})

    @pytest.fixture
    def string_data_with_nulls(self):
        """String Polars DataFrame with nulls."""
        return pl.DataFrame({
            "x": ["apple", None, "banana", "apple", "orange"],
            "y": [None, "cat", "dog", "cat", "dog"]
        }, schema={"x": pl.Utf8, "y": pl.Utf8})

    # --- Mean Strategy Tests ---
    def test_mean_strategy_fit(self, numeric_data_with_nulls):
        """Test fit with 'mean' strategy."""
        X = numeric_data_with_nulls
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X)

        assert imputer.statistics_ is not None
        # Mean of a: (1+2+4+5)/4 = 12/4 = 3.0
        # Mean of b: (10+30+50)/3 = 90/3 = 30.0
        # Mean of c: NaN, but Polars mean() on all-null float col is null.
        #            Imputer should handle this, e.g. by fill_null(0) or specific logic.
        #            Current SimpleImputer.fit for mean/median doesn't explicitly handle all-null column results from .mean()/.median()
        #            Polars X.mean() on an all-null Float64 column results in a null for that column.
        #            The transform will then try to fill nulls with null, which is a no-op.
        #            This means a column of all nulls, when strategy is mean/median, will remain all nulls.
        #            This is acceptable for this test; specific handling for all-null can be a feature.
        np.testing.assert_almost_equal(imputer.statistics_.select("a").item(), 3.0)
        np.testing.assert_almost_equal(imputer.statistics_.select("b").item(), 30.0)
        assert imputer.statistics_.select("c").item() is None

    def test_mean_strategy_transform(self, numeric_data_with_nulls):
        """Test transform with 'mean' strategy."""
        X = numeric_data_with_nulls
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X)
        X_transformed = imputer.transform(X)

        expected_a = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 5.0])
        expected_b = pl.Series("b", [10.0, 30.0, 30.0, 30.0, 50.0])
        expected_c = pl.Series("c", [None, None, None, None, None], dtype=pl.Float64) # Remains null

        assert X_transformed.select("a").to_series().equals(expected_a)
        assert X_transformed.select("b").to_series().equals(expected_b)
        assert X_transformed.select("c").to_series().equals(expected_c)


    def test_mean_strategy_non_numeric_error(self, string_data_with_nulls):
        """Test ValueError if 'mean' strategy is used with non-numeric data."""
        imputer = SimpleImputer(strategy='mean')
        with pytest.raises(ValueError, match="Strategy 'mean' can only be used with numeric data."):
            imputer.fit(string_data_with_nulls)

    # --- Median Strategy Tests ---
    def test_median_strategy_fit(self, numeric_data_with_nulls):
        """Test fit with 'median' strategy."""
        X = numeric_data_with_nulls
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X)

        assert imputer.statistics_ is not None
        # Median of a: [1,2,4,5] -> (2+4)/2 = 3.0
        # Median of b: [10,30,50] -> 30.0
        # Median of c: null
        np.testing.assert_almost_equal(imputer.statistics_.select("a").item(), 3.0)
        np.testing.assert_almost_equal(imputer.statistics_.select("b").item(), 30.0)
        assert imputer.statistics_.select("c").item() is None

    def test_median_strategy_transform(self, numeric_data_with_nulls):
        """Test transform with 'median' strategy."""
        X = numeric_data_with_nulls
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X)
        X_transformed = imputer.transform(X)

        expected_a = pl.Series("a", [1.0, 2.0, 3.0, 4.0, 5.0])
        expected_b = pl.Series("b", [10.0, 30.0, 30.0, 30.0, 50.0])
        expected_c = pl.Series("c", [None, None, None, None, None], dtype=pl.Float64)

        assert X_transformed.select("a").to_series().equals(expected_a)
        assert X_transformed.select("b").to_series().equals(expected_b)
        assert X_transformed.select("c").to_series().equals(expected_c)

    # --- Most Frequent Strategy Tests ---
    def test_most_frequent_strategy_numeric(self, numeric_data_with_nulls):
        """Test 'most_frequent' strategy with numeric data."""
        # For col 'a': all unique, mode might be first (1.0) or smallest. Polars mode returns all if counts are equal.
        # Let's make it clearer:
        X = pl.DataFrame({"a": [1.0, 2.0, 1.0, 4.0, 1.0, None]}) # Mode is 1.0
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(X)
        assert imputer.statistics_.select("a").item() == 1.0
        X_transformed = imputer.transform(X)
        expected_a = pl.Series("a", [1.0, 2.0, 1.0, 4.0, 1.0, 1.0])
        assert X_transformed.select("a").to_series().equals(expected_a)

    def test_most_frequent_strategy_string(self, string_data_with_nulls):
        """Test 'most_frequent' strategy with string data."""
        X = string_data_with_nulls # x: apple (2), banana (1), orange (1) -> apple
                                   # y: cat (2), dog (2) -> cat (polars mode picks first of ties)
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(X)
        assert imputer.statistics_.select("x").item() == "apple"
        assert imputer.statistics_.select("y").item() == "cat" # Assuming 'cat' comes before 'dog' in sorted unique if tie-breaking

        X_transformed = imputer.transform(X)
        expected_x = pl.Series("x", ["apple", "apple", "banana", "apple", "orange"])
        expected_y = pl.Series("y", ["cat", "cat", "dog", "cat", "dog"])
        assert X_transformed.select("x").to_series().equals(expected_x)
        assert X_transformed.select("y").to_series().equals(expected_y)

    def test_most_frequent_all_nulls_column(self):
        """Test 'most_frequent' with a column of all nulls."""
        X = pl.DataFrame({"a": [None, None, None]}, schema={"a": pl.Float64})
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(X)
        # Default fill for all-null numeric is 0
        assert imputer.statistics_.select("a").item() == 0
        X_transformed = imputer.transform(X)
        assert X_transformed.select("a").to_series().equals(pl.Series("a", [0.0, 0.0, 0.0]))

        X_str = pl.DataFrame({"b": [None, None]}, schema={"b": pl.Utf8})
        imputer_str = SimpleImputer(strategy='most_frequent')
        imputer_str.fit(X_str)
        # Default fill for all-null string is "missing"
        assert imputer_str.statistics_.select("b").item() == "missing"
        X_transformed_str = imputer_str.transform(X_str)
        assert X_transformed_str.select("b").to_series().equals(pl.Series("b",["missing", "missing"]))


    # --- Constant Strategy Tests ---
    def test_constant_strategy_numeric(self):
        """Test 'constant' strategy with a numeric fill_value."""
        X = pl.DataFrame({"a": [1.0, None, 3.0]}, schema={"a": pl.Float64})
        imputer = SimpleImputer(strategy='constant', fill_value=0.0)
        imputer.fit(X) # Fit is mostly for schema capture here
        assert imputer.fill_value_ == 0.0
        X_transformed = imputer.transform(X)
        expected_a = pl.Series("a", [1.0, 0.0, 3.0])
        assert X_transformed.select("a").to_series().equals(expected_a)

    def test_constant_strategy_string(self):
        """Test 'constant' strategy with a string fill_value."""
        X = pl.DataFrame({"x": ["hello", None, "world"]}, schema={"x": pl.Utf8})
        imputer = SimpleImputer(strategy='constant', fill_value="unknown")
        imputer.fit(X)
        assert imputer.fill_value_ == "unknown"
        X_transformed = imputer.transform(X)
        expected_x = pl.Series("x", ["hello", "unknown", "world"])
        assert X_transformed.select("x").to_series().equals(expected_x)

    def test_constant_strategy_default_fill_value(self, numeric_data_with_nulls, string_data_with_nulls):
        """Test 'constant' strategy with default fill_value."""
        X_num = numeric_data_with_nulls
        imputer_num = SimpleImputer(strategy='constant') # fill_value is None
        imputer_num.fit(X_num)
        # Default for numeric is 0
        assert imputer_num.statistics_.select("a").item() == 0
        assert imputer_num.statistics_.select("b").item() == 0
        assert imputer_num.statistics_.select("c").item() == 0
        X_transformed_num = imputer_num.transform(X_num)
        assert X_transformed_num.select("a").get_column("a")[2] == 0.0 # Check one imputed value

        X_str = string_data_with_nulls
        imputer_str = SimpleImputer(strategy='constant')
        imputer_str.fit(X_str)
        # Default for string is "missing_value"
        assert imputer_str.statistics_.select("x").item() == "missing_value"
        assert imputer_str.statistics_.select("y").item() == "missing_value"
        X_transformed_str = imputer_str.transform(X_str)
        assert X_transformed_str.select("x").get_column("x")[1] == "missing_value"


    # --- General and Error Tests ---
    def test_fit_empty_dataframe(self):
        """Test SimpleImputer fit with an empty DataFrame."""
        X_empty = pl.DataFrame(schema={"a": pl.Float64, "b": pl.Utf8})
        imputer = SimpleImputer(strategy='mean') # Mean will be problematic if transform non-empty
        imputer.fit(X_empty)
        assert imputer.n_features_in_ == 2
        assert imputer.feature_names_in_ == ["a", "b"]
        assert imputer.statistics_ is not None and imputer.statistics_.is_empty()

        # Transform on empty should also work
        X_transformed_empty = imputer.transform(X_empty)
        assert X_transformed_empty.is_empty()
        assert X_transformed_empty.columns == ["a", "b"]

    def test_transform_before_fit_error(self):
        """Test RuntimeError if transform is called before fit (for non-constant default)."""
        # For constant strategy with explicit fill_value, fit might not be strictly needed if schema is known.
        # However, current implementation requires fit to set up feature_names_in_ etc.
        # Let's test a strategy that definitely needs stats.
        imputer = SimpleImputer(strategy='mean')
        X = pl.DataFrame({"a": [1.0, None]})
        # This will fail because n_features_in_ is None.
        # The error message will be "SimpleImputer has not been fitted with valid statistics yet."
        # or similar from the n_features_in_ check
        with pytest.raises(RuntimeError, match="SimpleImputer has not been fitted with valid statistics yet"):
            imputer.transform(X)

    def test_inconsistent_features_error(self, numeric_data_with_nulls):
        """Test ValueError if transform is called with inconsistent features."""
        X_fit = numeric_data_with_nulls
        imputer = SimpleImputer()
        imputer.fit(X_fit)

        X_transform_wrong_n = pl.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError, match="features, but SimpleImputer was fitted with"):
            imputer.transform(X_transform_wrong_n)

        X_transform_wrong_names = pl.DataFrame({"d": [1.0], "e": [2.0], "f": [3.0]}) # Matches n_features but not names
        with pytest.raises(ValueError, match="Feature names or order of X do not match"):
            imputer.transform(X_transform_wrong_names)


    def test_invalid_strategy_error(self):
        """Test ValueError for invalid strategy."""
        with pytest.raises(ValueError, match="Strategy must be one of 'mean', 'median', 'most_frequent', or 'constant'."):
            SimpleImputer(strategy='unknown')

    def test_add_indicator_not_implemented(self):
        """Test NotImplementedError for add_indicator=True."""
        with pytest.raises(NotImplementedError, match="add_indicator=True is not yet implemented."):
            SimpleImputer(add_indicator=True)

    def test_no_nulls_passthrough(self):
        """Test that data with no nulls is unchanged."""
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        imputer_mean = SimpleImputer(strategy='mean')
        assert imputer_mean.fit_transform(X).equals(X)

        imputer_median = SimpleImputer(strategy='median')
        assert imputer_median.fit_transform(X).equals(X)

        imputer_mf = SimpleImputer(strategy='most_frequent')
        assert imputer_mf.fit_transform(X).equals(X)

        imputer_const = SimpleImputer(strategy='constant', fill_value=99.0)
        assert imputer_const.fit_transform(X).equals(X)

    def test_fit_transform_chaining(self, numeric_data_with_nulls):
        """Test that fit_transform works and returns self."""
        X = numeric_data_with_nulls
        imputer = SimpleImputer()
        X_transformed = imputer.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
        assert imputer.statistics_ is not None # Fit part should have run

    def test_different_missing_values_param_is_note(self):
        """Test that missing_values param is informational for Polars (as Polars uses nulls)."""
        # This test is more conceptual. Polars fill_null works on Polars nulls.
        # The `missing_values` parameter is standard in scikit-learn but less directly
        # used when the input must be a Polars DataFrame where null is the canonical missing value.
        # If we were to accept non-Polars nulls (e.g. a specific number like -1 or string "NA"),
        # we'd need a preprocessing step to convert those to Polars nulls first.
        X = pl.DataFrame({"a": [1, -1, 3]}, schema={"a":pl.Int64}).with_columns(
            pl.when(pl.col("a") == -1).then(None).otherwise(pl.col("a")).alias("a")
        ) # a: [1, null, 3]

        imputer = SimpleImputer(missing_values=-1, strategy="mean") # -1 is what we'd replace if not Polars
        # However, SimpleImputer currently works on Polars nulls.
        # This test just ensures it runs. The `missing_values` arg doesn't change Polars' null handling.
        imputer.fit(X)
        assert imputer.statistics_.item(0,0) == 2.0 # Mean of 1 and 3
        X_transformed = imputer.transform(X)
        assert X_transformed.get_column("a").to_list() == [1.0, 2.0, 3.0]

    def test_empty_df_fit_then_transform_non_empty_error(self):
        """Test that transforming non-empty data after fitting on empty raises error if no stats."""
        X_empty_schema = pl.DataFrame(schema={"a": pl.Float64, "b": pl.Float64})
        imputer_mean = SimpleImputer(strategy="mean")
        imputer_mean.fit(X_empty_schema)

        X_non_empty = pl.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
        with pytest.raises(RuntimeError, match="Imputer not fitted or statistics are not computed"):
            imputer_mean.transform(X_non_empty)

        # Constant strategy with explicit fill_value might be an exception if we decide it can work
        imputer_const_explicit = SimpleImputer(strategy="constant", fill_value=99)
        imputer_const_explicit.fit(X_empty_schema) # Fit sets up n_features_in_ etc.
        # fill_value_ is set. This should work.
        transformed = imputer_const_explicit.transform(X_non_empty)
        expected = pl.DataFrame({"a": [1.0, 99.0], "b": [99.0, 2.0]})
        assert transformed.equals(expected)

        imputer_const_default = SimpleImputer(strategy="constant") # Default fill value
        imputer_const_default.fit(X_empty_schema) # Fit on empty, statistics_ will be empty with schema
        with pytest.raises(RuntimeError, match="Imputer not fitted or fitted on empty data without default fill values determined"):
             imputer_const_default.transform(X_non_empty)


    def test_statistics_dtype_consistency(self):
        """Test that statistics_ dtypes are consistent with input."""
        X_int = pl.DataFrame({"a": [1, None, 3]}, schema={"a": pl.Int32})
        imputer_mean_int = SimpleImputer(strategy="mean")
        imputer_mean_int.fit(X_int)
        # Mean of integers can be float
        assert imputer_mean_int.statistics_["a"].dtype == pl.Float64

        X_float = pl.DataFrame({"a": [1.0, None, 3.0]}, schema={"a": pl.Float64})
        imputer_mean_float = SimpleImputer(strategy="mean")
        imputer_mean_float.fit(X_float)
        assert imputer_mean_float.statistics_["a"].dtype == pl.Float64

        # Most frequent should retain original dtype if possible
        X_cat = pl.DataFrame({"a": ["x", None, "y", "x"]}, schema={"a": pl.Categorical})
        imputer_mf_cat = SimpleImputer(strategy="most_frequent")
        imputer_mf_cat.fit(X_cat)
        # Polars mode() might return Utf8 if original was Categorical, then we cast back.
        # Current SimpleImputer casts mode to original dtype.
        assert imputer_mf_cat.statistics_["a"].dtype == pl.Categorical

        # Constant with explicit fill value
        imputer_const_int_fill = SimpleImputer(strategy="constant", fill_value=0)
        imputer_const_int_fill.fit(X_int) # Fit on Int32 data
        # statistics_ should reflect the data's original type for constant strategy
        assert imputer_const_int_fill.statistics_["a"].dtype == X_int["a"].dtype # Int32

        imputer_const_str_fill = SimpleImputer(strategy="constant", fill_value="test")
        imputer_const_str_fill.fit(X_cat) # Fit on Categorical data
        # This will create statistics_ with dtype Utf8 because fill_value is string.
        # The transform will then try to fill Categorical with Utf8, which might error or cast.
        # This highlights a subtle point: fill_value type vs column type.
        # Polars fill_null(value) will try to cast `value` to series dtype.
        # So if X_cat["a"] is Categorical, and we fill_null("test"), "test" is cast to Categorical.
        assert imputer_const_str_fill.statistics_["a"].dtype == X_cat["a"].dtype # Should be Categorical

        # Constant with default fill value
        imputer_const_default_int = SimpleImputer(strategy="constant")
        imputer_const_default_int.fit(X_int)
        assert imputer_const_default_int.statistics_["a"].dtype == X_int["a"].dtype # Int32, filled with 0

        imputer_const_default_str = SimpleImputer(strategy="constant")
        imputer_const_default_str.fit(X_cat)
        # Default for non-numeric is "missing_value" (Utf8), then cast to column type (Categorical)
        assert imputer_const_default_str.statistics_["a"].dtype == X_cat["a"].dtype # Categorical
