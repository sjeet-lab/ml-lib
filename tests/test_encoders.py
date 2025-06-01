"""Tests for categorical and label encoders."""

import numpy as np
import polars as pl
import pytest

from custom_ml_library.preprocessing import LabelEncoder, OneHotEncoder


class TestLabelEncoder:
    """Tests for the LabelEncoder class."""

    def test_fit_transform_simple(self):
        """Test basic fit and transform with strings."""
        y = pl.Series("labels", ["a", "b", "c", "a", "b", "b"])
        encoder = LabelEncoder()
        y_transformed = encoder.fit_transform(y)

        assert encoder.classes_ is not None
        np.testing.assert_array_equal(encoder.classes_, np.array(["a", "b", "c"]))
        expected_transformed = pl.Series("labels", [0, 1, 2, 0, 1, 1], dtype=pl.Int64)
        assert y_transformed.equals(expected_transformed)

    def test_transform_unseen_error(self):
        """Test ValueError when transform encounters unseen labels."""
        y_fit = pl.Series(["a", "b"])
        encoder = LabelEncoder()
        encoder.fit(y_fit)
        y_transform = pl.Series(["a", "c"])
        with pytest.raises(ValueError, match="y contains previously unseen label: c"):
            encoder.transform(y_transform)

    def test_inverse_transform(self):
        """Test inverse_transform back to original labels."""
        y_original = pl.Series("labels", ["cat", "dog", "mouse", "cat"])
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_original)
        y_decoded = encoder.inverse_transform(y_encoded)
        assert y_decoded.equals(y_original)
        assert y_decoded.dtype == y_original.dtype


    def test_inverse_transform_out_of_range_error(self):
        """Test ValueError for out-of-range labels in inverse_transform."""
        y_fit = pl.Series(["a", "b"]) # Encoded as 0, 1
        encoder = LabelEncoder()
        encoder.fit(y_fit)
        y_transform_invalid = pl.Series([0, 2], dtype=pl.Int64) # 2 is out of range
        with pytest.raises(ValueError, match="y contains values outside of the range"):
            encoder.inverse_transform(y_transform_invalid)

    def test_fit_empty_series_error(self):
        """Test ValueError when fitting on an empty Series."""
        y_empty = pl.Series("empty", [], dtype=pl.Utf8)
        encoder = LabelEncoder()
        with pytest.raises(ValueError, match="Cannot fit LabelEncoder with an empty Series."):
            encoder.fit(y_empty)

    def test_transform_not_fitted_error(self):
        """Test RuntimeError if transform is called before fit."""
        encoder = LabelEncoder()
        with pytest.raises(RuntimeError, match="LabelEncoder has not been fitted yet."):
            encoder.transform(pl.Series(["a"]))

    def test_inverse_transform_not_fitted_error(self):
        """Test RuntimeError if inverse_transform is called before fit."""
        encoder = LabelEncoder()
        with pytest.raises(RuntimeError, match="LabelEncoder has not been fitted yet."):
            encoder.inverse_transform(pl.Series([0]))

    def test_numeric_labels(self):
        """Test LabelEncoder with numeric labels."""
        y = pl.Series("numbers", [10, 20, 0, 10, 0, 0])
        encoder = LabelEncoder()
        y_transformed = encoder.fit_transform(y)

        np.testing.assert_array_equal(encoder.classes_, np.array([0, 10, 20]))
        expected_transformed = pl.Series("numbers", [1, 2, 0, 1, 0, 0], dtype=pl.Int64)
        assert y_transformed.equals(expected_transformed)

        y_decoded = encoder.inverse_transform(y_transformed)
        assert y_decoded.equals(y)
        assert y_decoded.dtype == y.dtype

    def test_inverse_transform_empty_series(self):
        """Test inverse_transform on an empty Series after fitting."""
        y_fit = pl.Series(["a", "b", "c"])
        encoder = LabelEncoder()
        encoder.fit(y_fit)

        y_empty_encoded = pl.Series("empty_enc", [], dtype=pl.Int64)
        y_empty_decoded = encoder.inverse_transform(y_empty_encoded)

        assert y_empty_decoded.is_empty()
        assert y_empty_decoded.dtype == y_fit.dtype # Should match original fitted dtype


class TestOneHotEncoder:
    """Tests for the OneHotEncoder class."""

    @pytest.fixture
    def sample_ohe_data(self):
        """Sample data for OneHotEncoder tests."""
        return pl.DataFrame({
            "cat_feature_1": [0, 1, 0, 2, 1],
            "cat_feature_2": ["a", "b", "a", "c", "b"]
        })

    def test_fit_and_transform_basic(self, sample_ohe_data):
        """Test basic fit and transform."""
        X = sample_ohe_data
        ohe = OneHotEncoder()
        X_transformed = ohe.fit_transform(X)

        assert ohe.n_features_in_ == 2
        assert ohe.feature_names_in_ == ["cat_feature_1", "cat_feature_2"]

        # Check categories_
        assert len(ohe.categories_) == 2
        np.testing.assert_array_equal(ohe.categories_[0], np.array([0, 1, 2]))
        np.testing.assert_array_equal(ohe.categories_[1], np.array(["a", "b", "c"]))

        # Check output feature names
        expected_feature_names = [
            "cat_feature_1_0", "cat_feature_1_1", "cat_feature_1_2",
            "cat_feature_2_a", "cat_feature_2_b", "cat_feature_2_c"
        ]
        assert ohe.get_feature_names_out() == expected_feature_names
        assert X_transformed.columns == expected_feature_names

        # Check a few transformed values
        # Row 0: [0, "a"] -> [1,0,0, 1,0,0]
        expected_row_0 = [1,0,0, 1,0,0]
        np.testing.assert_array_equal(X_transformed.row(0), tuple(expected_row_0))
        # Row 1: [1, "b"] -> [0,1,0, 0,1,0]
        expected_row_1 = [0,1,0, 0,1,0]
        np.testing.assert_array_equal(X_transformed.row(1), tuple(expected_row_1))

        # All values should be UInt8 (0 or 1)
        for col_name in X_transformed.columns:
            assert X_transformed[col_name].dtype == pl.UInt8

    def test_handle_unknown_error(self, sample_ohe_data):
        """Test handle_unknown='error' raises ValueError."""
        X_fit = sample_ohe_data
        ohe = OneHotEncoder(handle_unknown='error')
        ohe.fit(X_fit)

        X_transform_unknown = pl.DataFrame({
            "cat_feature_1": [0, 3], # 3 is unknown
            "cat_feature_2": ["a", "d"]  # "d" is unknown
        })
        with pytest.raises(ValueError, match="Found unknown categories .* in column 'cat_feature_1'"):
            ohe.transform(X_transform_unknown)

    def test_handle_unknown_ignore(self, sample_ohe_data):
        """Test handle_unknown='ignore' produces all zeros for unknown categories."""
        X_fit = sample_ohe_data
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(X_fit)

        X_transform_unknown = pl.DataFrame({
            "cat_feature_1": [0, 3, 1], # 3 is unknown
            "cat_feature_2": ["a", "d", "b"]  # "d" is unknown
        })
        X_transformed = ohe.transform(X_transform_unknown)

        # Row 0: [0, "a"] -> [1,0,0, 1,0,0] (known)
        expected_row_0 = [1,0,0, 1,0,0]
        np.testing.assert_array_equal(X_transformed.row(0), tuple(expected_row_0))

        # Row 1: [3, "d"] (both unknown) -> [0,0,0, 0,0,0]
        expected_row_1 = [0,0,0, 0,0,0]
        np.testing.assert_array_equal(X_transformed.row(1), tuple(expected_row_1))

        # Row 2: [1, "b"] -> [0,1,0, 0,1,0] (known)
        expected_row_2 = [0,1,0, 0,1,0]
        np.testing.assert_array_equal(X_transformed.row(2), tuple(expected_row_2))


    def test_fit_empty_dataframe_error(self):
        """Test fitting on an empty DataFrame (with schema)."""
        # Current OneHotEncoder.fit allows empty DF with schema,
        # it fits empty categories and generates no output feature names.
        X_empty_schema = pl.DataFrame(schema={"a":pl.Int64, "b":pl.Utf8})
        ohe = OneHotEncoder()
        ohe.fit(X_empty_schema)

        assert ohe.n_features_in_ == 2
        assert ohe.feature_names_in_ == ["a","b"]
        assert len(ohe.categories_) == 2
        assert len(ohe.categories_[0]) == 0 # No categories learned
        assert len(ohe.categories_[1]) == 0
        assert ohe.get_feature_names_out() == []

        # Transform on empty should also work and produce empty DF with no columns if no features out
        X_transformed_empty = ohe.transform(X_empty_schema.clone())
        assert X_transformed_empty.is_empty()
        assert X_transformed_empty.columns == []


    def test_transform_not_fitted_error(self):
        """Test RuntimeError if transform is called before fit."""
        ohe = OneHotEncoder()
        with pytest.raises(RuntimeError, match="OneHotEncoder has not been fitted yet."):
            ohe.transform(pl.DataFrame({"a": [1]}))

    def test_inconsistent_features_transform(self, sample_ohe_data):
        """Test ValueError for inconsistent features during transform."""
        ohe = OneHotEncoder()
        ohe.fit(sample_ohe_data)

        X_wrong_num_feat = pl.DataFrame({"c1":[1]})
        with pytest.raises(ValueError, match="features, but OneHotEncoder was fitted with"):
            ohe.transform(X_wrong_num_feat)

        X_wrong_names = pl.DataFrame({"feature_A": [0], "feature_B": ["x"]}) # Same num but wrong names
        with pytest.raises(ValueError, match="Feature names or order of X do not match"):
            ohe.transform(X_wrong_names)

    # TODO: Add tests for 'drop' parameter once implemented.
    # def test_drop_first(self): pass
    # def test_drop_if_binary(self): pass

    def test_get_feature_names_out_not_fitted(self):
        """Test get_feature_names_out before fit."""
        ohe = OneHotEncoder()
        with pytest.raises(RuntimeError, match="OneHotEncoder has not been fitted yet."):
            ohe.get_feature_names_out()

    def test_single_feature_ohe(self):
        """Test OneHotEncoder with a single feature."""
        X = pl.DataFrame({"feat": ["x", "y", "x"]})
        ohe = OneHotEncoder()
        X_transformed = ohe.fit_transform(X)

        expected_names = ["feat_x", "feat_y"]
        assert ohe.get_feature_names_out() == expected_names
        assert X_transformed.columns == expected_names

        # Row 0: "x" -> [1,0]
        np.testing.assert_array_equal(X_transformed.row(0), (1,0))
        # Row 1: "y" -> [0,1]
        np.testing.assert_array_equal(X_transformed.row(1), (0,1))

    def test_categories_with_nulls_in_fit(self):
        """Test how nulls are handled during category fitting."""
        X = pl.DataFrame({"a": [1, None, 2, 1]}, schema={"a": pl.Int64})
        ohe = OneHotEncoder()
        ohe.fit(X) # drop_nulls() is used in fit for categories
        np.testing.assert_array_equal(ohe.categories_[0], np.array([1,2]))

        # Transform should work, nulls in input X won't match any category
        # and will result in all-zero rows for that feature's one-hot columns
        # if handle_unknown='ignore' (implicitly, as null is not a category).
        # If handle_unknown='error', it depends if null is considered "unknown".
        # Current implementation: nulls in X become 0 for all categories of that feature.
        X_transformed = ohe.transform(X)
        # X[1] was None -> a_1=0, a_2=0
        assert X_transformed.row(1) == (0,0) # (a_1, a_2)
        # X[0] was 1 -> a_1=1, a_2=0
        assert X_transformed.row(0) == (1,0)

    def test_transform_empty_df_fitted_on_data(self, sample_ohe_data):
        """Test transforming an empty DataFrame after fitting on data."""
        ohe = OneHotEncoder()
        ohe.fit(sample_ohe_data)

        X_empty_correct_schema = pl.DataFrame(schema=sample_ohe_data.schema)
        X_transformed_empty = ohe.transform(X_empty_correct_schema)

        assert X_transformed_empty.is_empty()
        assert X_transformed_empty.columns == ohe.get_feature_names_out()
        for col_name in X_transformed_empty.columns:
            assert X_transformed_empty[col_name].dtype == pl.UInt8

    def test_fit_on_empty_df_no_cols(self):
        """Test fitting on an empty DataFrame with no columns."""
        X_empty_no_cols = pl.DataFrame()
        ohe = OneHotEncoder()
        ohe.fit(X_empty_no_cols)
        assert ohe.n_features_in_ == 0
        assert ohe.feature_names_in_ == []
        assert ohe.categories_ == []
        assert ohe.get_feature_names_out() == []

        X_transformed = ohe.transform(X_empty_no_cols.clone())
        assert X_transformed.is_empty()
        assert X_transformed.columns == []

    def test_transform_empty_df_fitted_on_empty_no_cols(self):
        """Test transform empty after fit on empty with no columns."""
        X_empty_no_cols = pl.DataFrame()
        ohe = OneHotEncoder()
        ohe.fit(X_empty_no_cols)
        X_transformed = ohe.transform(X_empty_no_cols.clone())
        assert X_transformed.is_empty()
        assert X_transformed.columns == []
        assert ohe.get_feature_names_out() == []
