"""Tests for preprocessing scalers."""

import numpy as np
import polars as pl
import pytest

from custom_ml_library.preprocessing import (
    MinMaxScaler,
    RobustScaler, # Added RobustScaler
    StandardScaler,
)


class TestStandardScaler:
    """Tests for the StandardScaler class."""

    def test_fit_and_transform(self):
        """Tests fit and transform methods separately."""
        X_pl = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        scaler = StandardScaler()
        scaler.fit(X_pl)

        # Check mean_ and scale_
        expected_mean_a = 3.0
        # expected_std_a = np.std([1.0, 2.0, 3.0, 4.0, 5.0])  # Unused
        expected_mean_b = 3.0
        # expected_std_b = np.std([5.0, 4.0, 3.0, 2.0, 1.0]) # Unused

        np.testing.assert_almost_equal(scaler.mean_.select("a").item(), expected_mean_a)
        np.testing.assert_almost_equal(
            scaler.scale_.select("a").item(), X_pl.select("a").std().item()
        )
        np.testing.assert_almost_equal(scaler.mean_.select("b").item(), expected_mean_b)
        np.testing.assert_almost_equal(
            scaler.scale_.select("b").item(), X_pl.select("b").std().item()
        )

        X_transformed_pl = scaler.transform(X_pl)
        X_transformed_np = X_transformed_pl.to_numpy()

        # Check transformed data properties (mean approx 0, std approx 1)
        np.testing.assert_array_almost_equal(
            X_transformed_np.mean(axis=0), [0.0, 0.0], decimal=6
        )
        np.testing.assert_array_almost_equal(
            X_transformed_np.std(axis=0, ddof=1),
            [1.0, 1.0],
            decimal=6,  # Use ddof=1 for sample std dev
        )

    def test_fit_transform(self):
        """Tests the fit_transform method."""
        X_pl = pl.DataFrame(
            {
                "a": [10.0, 20.0, 30.0, 40.0, 50.0],
                "b": [15.0, 25.0, 35.0, 45.0, 55.0],
            }
        )
        scaler = StandardScaler()
        X_transformed_pl = scaler.fit_transform(X_pl)
        X_transformed_np = X_transformed_pl.to_numpy()

        np.testing.assert_array_almost_equal(
            X_transformed_np.mean(axis=0), [0.0, 0.0], decimal=6
        )
        np.testing.assert_array_almost_equal(
            X_transformed_np.std(axis=0, ddof=1),
            [1.0, 1.0],
            decimal=6,  # Use ddof=1 for sample std dev
        )

        # Check that fit attributes are set
        assert scaler.mean_ is not None
        assert scaler.scale_ is not None

    def test_column_with_zero_std_dev(self):
        """Tests StandardScaler with a column of all same values."""
        X_pl = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 5.0, 5.0, 5.0, 5.0],  # Zero std dev
            }
        )
        scaler = StandardScaler()
        scaler.fit(X_pl)

        # scale_ for column 'b' should be 1.0
        np.testing.assert_almost_equal(scaler.scale_.select("b").item(), 1.0)

        X_transformed_pl = scaler.transform(X_pl)

        # Column 'b' transformed should be all zeros if mean is subtracted and scale is 1
        # (X - mean) / 1 = X - 5.0
        expected_b_transformed = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(
            X_transformed_pl.select("b").to_numpy().flatten(),
            expected_b_transformed,
            decimal=6,
        )

        # Check mean and std of transformed column 'b'
        # Mean should be 0, std should be 0 (since all values are 0)
        np.testing.assert_almost_equal(
            X_transformed_pl.select("b").mean().item(), 0.0, decimal=6
        )
        np.testing.assert_almost_equal(
            X_transformed_pl.select("b").std().item(), 0.0, decimal=6
        )

    def test_transform_before_fit_raises_error(self):
        """Tests that transform before fit raises a RuntimeError."""
        scaler = StandardScaler()
        X_pl = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError, match="Scaler has not been fitted yet."):
            scaler.transform(X_pl)


class TestMinMaxScaler:
    """Tests for the MinMaxScaler class."""

    def test_fit_and_transform_default_range(self):
        """Tests fit and transform with default feature_range (0, 1)."""
        X_pl = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],  # min=1, max=5, range=4
                "b": [
                    -10.0,
                    0.0,
                    10.0,
                    -5.0,
                    5.0,
                ],  # Padded to length 5. min=-10, max=10, range=20
            }
        )
        scaler = MinMaxScaler()
        scaler.fit(X_pl)

        # Expected:
        # col_a: data_min=1, data_max=5, data_range=4
        #        scale = (1-0)/4 = 0.25
        #        min = 0 - 1 * 0.25 = -0.25
        # col_b: data_min=-10, data_max=10, data_range=20
        #        scale = (1-0)/20 = 0.05
        #        min = 0 - (-10) * 0.05 = 0.5

        np.testing.assert_almost_equal(scaler.scale_.select("a").item(), 0.25)
        np.testing.assert_almost_equal(scaler.min_.select("a").item(), -0.25)
        np.testing.assert_almost_equal(scaler.scale_.select("b").item(), 0.05)
        np.testing.assert_almost_equal(scaler.min_.select("b").item(), 0.5)

        X_transformed_pl = scaler.transform(X_pl)
        X_transformed_np = X_transformed_pl.to_numpy()

        # Check if data is scaled to [0, 1]
        np.testing.assert_array_almost_equal(
            X_transformed_np.min(axis=0), [0.0, 0.0], decimal=6
        )
        np.testing.assert_array_almost_equal(
            X_transformed_np.max(axis=0), [1.0, 1.0], decimal=6
        )

    def test_fit_transform_custom_range(self):
        """Tests fit_transform with a custom feature_range, e.g., (-1, 1)."""
        X_pl = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],  # min=1, max=5, range=4
                "b": [-10.0, 0.0, 10.0, -5.0, 5.0],  # Padded. min=-10, max=10, range=20
            }
        )
        feature_range = (-1, 1)
        scaler = MinMaxScaler(feature_range=feature_range)

        # Expected for col_a with range (-1, 1):
        # data_min=1, data_max=5, data_range=4
        # scale = (1 - (-1)) / 4 = 2 / 4 = 0.5
        # min = -1 - 1 * 0.5 = -1.5
        # For col_b:
        # data_min=-10, data_max=10, data_range=20
        # scale = (1 - (-1)) / 20 = 2 / 20 = 0.1
        # min = -1 - (-10) * 0.1 = -1 + 1 = 0

        X_transformed_pl = scaler.fit_transform(X_pl)
        X_transformed_np = X_transformed_pl.to_numpy()

        np.testing.assert_almost_equal(scaler.scale_.select("a").item(), 0.5)
        np.testing.assert_almost_equal(scaler.min_.select("a").item(), -1.5)
        np.testing.assert_almost_equal(scaler.scale_.select("b").item(), 0.1)
        np.testing.assert_almost_equal(scaler.min_.select("b").item(), 0.0)

        # Check if data is scaled to [-1, 1]
        np.testing.assert_array_almost_equal(
            X_transformed_np.min(axis=0), [-1.0, -1.0], decimal=6
        )
        np.testing.assert_array_almost_equal(
            X_transformed_np.max(axis=0), [1.0, 1.0], decimal=6
        )

        assert scaler.feature_range == feature_range

    def test_column_with_zero_range(self):
        """Tests MinMaxScaler with a column of all same values (range is 0)."""
        X_pl = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 5.0, 5.0, 5.0, 5.0],  # Zero range
            }
        )
        scaler = MinMaxScaler(feature_range=(0, 1))  # Default range
        scaler.fit(X_pl)

        # scale_ for column 'b' should be 1.0
        np.testing.assert_almost_equal(scaler.scale_.select("b").item(), 1.0)
        # min_ for column 'b' should be feature_range[0] - data_min * 1.0 = 0 - 5.0 * 1.0 = -5.0
        np.testing.assert_almost_equal(scaler.min_.select("b").item(), -5.0)

        X_transformed_pl = scaler.transform(X_pl)
        # Expected for col b: X * 1.0 + (-5.0) = 5.0 * 1.0 - 5.0 = 0.0
        expected_b_transformed = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(
            X_transformed_pl.select("b").to_numpy().flatten(),
            expected_b_transformed,
            decimal=6,
        )

        # If feature_range is (0,1) and a column has zero range,
        # the transformed values for that column should all be 0.
        # This is because X * 1 + (0 - data_min * 1) = data_min - data_min = 0.
        # This isn't always ideal; some libraries map to feature_range[0].
        # Let's check the actual behavior based on the implementation.
        # If scale is 1, transform is X * 1 + (min_range - data_min * 1)
        # For col b: 5.0 * 1.0 + (0 - 5.0 * 1.0) = 5.0 - 5.0 = 0.0. Correct.

        scaler_custom_range = MinMaxScaler(feature_range=(10, 20))
        scaler_custom_range.fit(X_pl)  # data_min for b is 5.0
        # scale_ for 'b' should be 1.0
        np.testing.assert_almost_equal(
            scaler_custom_range.scale_.select("b").item(), 1.0
        )
        # min_ for 'b' = 10 - 5.0 * 1.0 = 5.0
        np.testing.assert_almost_equal(scaler_custom_range.min_.select("b").item(), 5.0)

        X_transformed_custom_pl = scaler_custom_range.transform(X_pl)
        # For col b: X * 1.0 + 5.0 = 5.0 * 1.0 + 5.0 = 10.0
        # This means all values in the zero-range column are mapped to the min of the feature_range.
        expected_b_custom_transformed = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

        np.testing.assert_array_almost_equal(
            X_transformed_custom_pl.select("b").to_numpy().flatten(),
            expected_b_custom_transformed,
            decimal=6,
        )

    def test_transform_before_fit_raises_error(self):
        """Tests that transform before fit raises a RuntimeError."""
        scaler = MinMaxScaler()
        X_pl = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError, match="Scaler has not been fitted yet."):
            scaler.transform(X_pl)

    def test_single_feature_dataframe(self):
        """Tests MinMaxScaler with a single feature DataFrame."""
        X_pl = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaler = MinMaxScaler()
        X_transformed_pl = scaler.fit_transform(X_pl)
        X_transformed_np = X_transformed_pl.to_numpy().flatten()

        np.testing.assert_almost_equal(X_transformed_np.min(), 0.0, decimal=6)
        np.testing.assert_almost_equal(X_transformed_np.max(), 1.0, decimal=6)

        # Check attributes are Polars Series like
        assert isinstance(
            scaler.min_, pl.DataFrame
        )  # In current impl, these are DataFrames
        assert isinstance(scaler.scale_, pl.DataFrame)
        assert scaler.min_.shape == (1, 1)
        assert scaler.scale_.shape == (1, 1)
        np.testing.assert_almost_equal(scaler.scale_.item(), 0.25)  # (1-0)/(5-1)
        np.testing.assert_almost_equal(scaler.min_.item(), -0.25)  # 0 - 1*0.25

    def test_fit_transform_output_type(self):
        """Tests that fit_transform returns a Polars DataFrame."""
        X_pl = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = MinMaxScaler()
        X_transformed = scaler.fit_transform(X_pl)
        assert isinstance(X_transformed, pl.DataFrame)
        assert X_transformed.columns == ["a"]

    def test_standard_scaler_output_type(self):
        """Tests that StandardScaler fit_transform returns a Polars DataFrame."""
        X_pl = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(X_pl)
        assert isinstance(X_transformed, pl.DataFrame)
        assert X_transformed.columns == ["a"]


# RobustScaler Tests
class TestRobustScaler:
    """Tests for the RobustScaler class."""

    @pytest.fixture
    def sample_data_rs(self):
        """Sample data for RobustScaler tests."""
        return pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0], # Outlier
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0], # Constant part, then some same values
            "c": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Constant
        })

    def test_fit_default(self, sample_data_rs):
        """Test fit with default parameters."""
        X = sample_data_rs
        scaler = RobustScaler()
        scaler.fit(X)

        assert scaler.n_features_in_ == X.shape[1]
        assert scaler.feature_names_in_ == X.columns
        assert scaler.center_ is not None
        assert scaler.scale_ is not None

        # Expected median for 'a': (5.0 + 6.0) / 2 = 5.5
        # Expected Q1 for 'a': 3.0 (25th percentile of 1..9, 100) -> (1,2,3,4,5,6,7,8,9,100) -> values[2] = 3
        # Expected Q3 for 'a': 8.0 (75th percentile of 1..9, 100) -> values[7] = 8
        # Expected IQR for 'a': 8.0 - 3.0 = 5.0
        np.testing.assert_almost_equal(scaler.center_.select("a").item(), 5.5)
        np.testing.assert_almost_equal(scaler.scale_.select("a").item(), 5.0)

        # Expected median for 'b': 50.0
        # Expected Q1 for 'b': 30.0 (values are 10,20,30,40,50,50,50,50,50,50) -> values[2] = 30
        # Expected Q3 for 'b': 50.0 (values[7]=50)
        # Expected IQR for 'b': 50.0 - 30.0 = 20.0
        np.testing.assert_almost_equal(scaler.center_.select("b").item(), 50.0)
        np.testing.assert_almost_equal(scaler.scale_.select("b").item(), 20.0)

        # Expected median for 'c': 1.0
        # Expected Q1 for 'c': 1.0
        # Expected Q3 for 'c': 1.0
        # Expected IQR for 'c': 0.0, so scale_ should be 1.0
        np.testing.assert_almost_equal(scaler.center_.select("c").item(), 1.0)
        np.testing.assert_almost_equal(scaler.scale_.select("c").item(), 1.0)


    def test_transform_default(self, sample_data_rs):
        """Test transform with default parameters."""
        X = sample_data_rs
        scaler = RobustScaler()
        scaler.fit(X)
        X_transformed = scaler.transform(X)

        assert X_transformed.shape == X.shape
        assert X_transformed.columns == X.columns

        # For col 'a': (X_a - 5.5) / 5.0
        # X_a[0] = 1.0 -> (1.0 - 5.5) / 5.0 = -4.5 / 5.0 = -0.9
        # X_a[9] = 100.0 -> (100.0 - 5.5) / 5.0 = 94.5 / 5.0 = 18.9
        np.testing.assert_almost_equal(X_transformed.select("a").row(0)[0], -0.9)
        np.testing.assert_almost_equal(X_transformed.select("a").row(9)[0], 18.9)

        # For col 'c' (constant): (X_c - 1.0) / 1.0 = 0.0 for all
        assert (X_transformed.select("c") == 0.0).all().item()


    def test_fit_transform_no_centering(self, sample_data_rs):
        """Test fit_transform with no centering."""
        X = sample_data_rs
        scaler = RobustScaler(with_centering=False)
        X_transformed = scaler.fit_transform(X)

        assert scaler.center_ is None
        assert scaler.scale_ is not None
        # For col 'a': X_a / 5.0 (scale is IQR = 5.0)
        # X_a[0] = 1.0 -> 1.0 / 5.0 = 0.2
        np.testing.assert_almost_equal(X_transformed.select("a").row(0)[0], 0.2)

    def test_fit_transform_no_scaling(self, sample_data_rs):
        """Test fit_transform with no scaling."""
        X = sample_data_rs
        scaler = RobustScaler(with_scaling=False)
        X_transformed = scaler.fit_transform(X)

        assert scaler.center_ is not None
        assert scaler.scale_ is None
        # For col 'a': X_a - 5.5 (center is median = 5.5)
        # X_a[0] = 1.0 -> 1.0 - 5.5 = -4.5
        np.testing.assert_almost_equal(X_transformed.select("a").row(0)[0], -4.5)

    def test_fit_transform_no_centering_no_scaling(self, sample_data_rs):
        """Test fit_transform with no centering and no scaling (should be a no-op)."""
        X = sample_data_rs
        scaler = RobustScaler(with_centering=False, with_scaling=False)
        X_transformed = scaler.fit_transform(X)
        assert scaler.center_ is None
        assert scaler.scale_ is None
        assert X_transformed.equals(X) # Should be identical

    def test_different_quantile_range(self):
        """Test with a different quantile range."""
        X = pl.DataFrame({"a": np.arange(1, 101).astype(float)}) # 1 to 100
        # q_min=10 (10th value), q_max=90 (90th value)
        # For 1..100, 10th percentile is around 10.0, 90th percentile is around 90.0
        # Median is (50+51)/2 = 50.5
        # Scale should be 90.0 - 10.0 = 80.0 (approximately, depends on interpolation)
        # Polars quantile with strategy='linear':
        # q(0.1) = 1 + (100-1)*0.1 = 1 + 9.9 = 10.9 (or value at index for 'nearest')
        # Polars default interpolation is 'linear'
        # For X.quantile(0.1, interpolation='linear'): 10.9
        # For X.quantile(0.9, interpolation='linear'): 90.1
        # IQR = 90.1 - 10.9 = 79.2
        # Median = 50.5

        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(X)

        expected_median = 50.5
        # For Polars default 'linear' interpolation:
        # q10 = 1 + (100-1)*0.1 = 10.9.
        # q90 = 1 + (100-1)*0.9 = 90.1.
        # iqr = 90.1 - 10.9 = 79.2
        expected_q10 = X.quantile(0.10, interpolation="linear").item(0,0)
        expected_q90 = X.quantile(0.90, interpolation="linear").item(0,0)
        expected_iqr = expected_q90 - expected_q10

        np.testing.assert_almost_equal(scaler.center_.item(0,0), expected_median)
        np.testing.assert_almost_equal(scaler.scale_.item(0,0), expected_iqr)

    def test_fit_on_empty_df(self):
        """Test fitting on an empty DataFrame."""
        X_empty = pl.DataFrame({"a": [], "b": []}, schema={"a":pl.Float64, "b":pl.Float64})
        scaler = RobustScaler()
        scaler.fit(X_empty)
        assert scaler.n_features_in_ == 2
        assert scaler.feature_names_in_ == ["a", "b"]
        assert scaler.center_.is_empty() # Center should be an empty DF with original schema
        assert scaler.scale_.is_empty()  # Scale should be an empty DF with original schema

        # Transform on empty should also work
        X_transformed_empty = scaler.transform(X_empty)
        assert X_transformed_empty.is_empty()
        assert X_transformed_empty.columns == ["a", "b"]

    def test_transform_on_empty_fitted_on_data(self, sample_data_rs):
        """Test transforming an empty DataFrame after fitting on data."""
        scaler = RobustScaler()
        scaler.fit(sample_data_rs)
        X_empty_correct_schema = pl.DataFrame(schema=sample_data_rs.schema)
        X_transformed_empty = scaler.transform(X_empty_correct_schema)
        assert X_transformed_empty.is_empty()
        assert X_transformed_empty.columns == sample_data_rs.columns


    def test_error_transform_before_fit(self):
        """Test RuntimeError if transform is called before fit."""
        scaler = RobustScaler()
        X = pl.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="Scaler has not been fitted yet"):
            scaler.transform(X)

    def test_error_inconsistent_features_transform(self, sample_data_rs):
        """Test ValueError if X in transform has different number of features."""
        X_fit = sample_data_rs
        scaler = RobustScaler()
        scaler.fit(X_fit)
        X_transform_wrong_n_features = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        with pytest.raises(ValueError, match="features, but RobustScaler was fitted with"):
            scaler.transform(X_transform_wrong_n_features)

        X_transform_wrong_names = pl.DataFrame({"d": [1.0], "e": [2.0], "f": [3.0]})
        with pytest.raises(ValueError, match="Feature names or order of X do not match"):
            scaler.transform(X_transform_wrong_names)


    def test_invalid_quantile_range(self):
        """Test ValueError for invalid quantile_range."""
        with pytest.raises(ValueError, match="Invalid quantile_range"):
            RobustScaler(quantile_range=(75.0, 25.0)) # q_min > q_max
        with pytest.raises(ValueError, match="Invalid quantile_range"):
            RobustScaler(quantile_range=(-10.0, 25.0)) # q_min < 0
        with pytest.raises(ValueError, match="Invalid quantile_range"):
            RobustScaler(quantile_range=(25.0, 110.0)) # q_max > 100
        with pytest.raises(ValueError, match="Invalid quantile_range"):
            RobustScaler(quantile_range=(25.0, 25.0)) # q_min == q_max

    def test_single_column_dataframe(self):
        """Test RobustScaler with a single column DataFrame."""
        X = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 10.0]}) # Median 3, Q1=2, Q3=4, IQR=2
        scaler = RobustScaler()
        X_transformed = scaler.fit_transform(X)

        np.testing.assert_almost_equal(scaler.center_.item(0,0), 3.0)
        np.testing.assert_almost_equal(scaler.scale_.item(0,0), 2.0)

        # (1-3)/2 = -1
        # (2-3)/2 = -0.5
        # (3-3)/2 = 0
        # (4-3)/2 = 0.5
        # (10-3)/2 = 3.5
        expected_transformed = pl.DataFrame({"a": [-1.0, -0.5, 0.0, 0.5, 3.5]})
        assert X_transformed.equals(expected_transformed, check_dtype=False) # Check_dtype false due to potential float precision
