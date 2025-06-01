"""Tests for preprocessing scalers."""

import numpy as np
import polars as pl
import pytest

from custom_ml_library.preprocessing import StandardScaler, MinMaxScaler


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
