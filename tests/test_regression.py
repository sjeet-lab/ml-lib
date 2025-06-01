"""Tests for regression models."""

import numpy as np
import polars as pl
import pytest

from regression.linear_regression import LinearRegression


class TestLinearRegression:
  """Tests for the LinearRegression class."""

  def test_fit_intercept_true(self):
    """Tests LinearRegression with fit_intercept=True."""
    # y = 2*X1 + 3*X2 + 5
    X_pl = pl.DataFrame({
        "X1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "X2": [2.0, 3.0, 2.5, 4.5, 5.0],
    })
    # y = 2*X1 + 3*X2 + 5
    y_np = 2 * X_pl.select("X1").to_series().to_numpy() + \
           3 * X_pl.select("X2").to_series().to_numpy() + 5
    y_pl = pl.Series("y", y_np)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_pl, y_pl)

    expected_coefs = np.array([2.0, 3.0])
    expected_intercept = 5.0

    np.testing.assert_array_almost_equal(model.coef_, expected_coefs, decimal=5)
    np.testing.assert_almost_equal(model.intercept_, expected_intercept, decimal=5)

    predictions_pl = model.predict(X_pl)
    np.testing.assert_array_almost_equal(
        predictions_pl.to_numpy(), y_pl.to_numpy(), decimal=5
    )
    assert isinstance(predictions_pl, pl.Series)

  def test_fit_intercept_false(self):
    """Tests LinearRegression with fit_intercept=False."""
    # y = 2*X1 + 3*X2
    X_pl = pl.DataFrame({
        "X1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "X2": [2.0, 3.0, 2.5, 4.5, 5.0],
    })
    y_np = 2 * X_pl.select("X1").to_series().to_numpy() + \
           3 * X_pl.select("X2").to_series().to_numpy()
    y_pl = pl.Series("y", y_np)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_pl, y_pl)

    expected_coefs = np.array([2.0, 3.0])
    expected_intercept = 0.0

    np.testing.assert_array_almost_equal(model.coef_, expected_coefs, decimal=5)
    np.testing.assert_almost_equal(model.intercept_, expected_intercept, decimal=5)

    predictions_pl = model.predict(X_pl)
    np.testing.assert_array_almost_equal(
        predictions_pl.to_numpy(), y_pl.to_numpy(), decimal=5
    )

  def test_single_feature(self):
    """Tests LinearRegression with a single feature."""
    # y = 3*X1 + 2
    X_pl = pl.DataFrame({"X1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_np = 3 * X_pl.select("X1").to_series().to_numpy() + 2
    y_pl = pl.Series("y", y_np)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_pl, y_pl)

    expected_coefs = np.array([3.0])
    expected_intercept = 2.0

    np.testing.assert_array_almost_equal(model.coef_, expected_coefs, decimal=5)
    np.testing.assert_almost_equal(model.intercept_, expected_intercept, decimal=5)

    predictions_pl = model.predict(X_pl)
    np.testing.assert_array_almost_equal(
        predictions_pl.to_numpy(), y_pl.to_numpy(), decimal=5
    )

  def test_predict_before_fit_raises_error(self):
    """Tests that predict before fit raises a RuntimeError."""
    model = LinearRegression()
    X_pl = pl.DataFrame({"X1": [1.0, 2.0, 3.0]})
    with pytest.raises(RuntimeError, match="This LinearRegression instance is not fitted yet."):
      model.predict(X_pl)

  def test_collinear_features(self):
    """Tests LinearRegression with perfectly collinear features."""
    # y = 2*X1 + 5 (X2 is a linear combination of X1)
    X_pl = pl.DataFrame({
        "X1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "X2": [2.0, 4.0, 6.0, 8.0, 10.0], # X2 = 2 * X1
    })
    # Due to perfect collinearity, the specific coefficients for X1 and X2 are
    # not uniquely identifiable, but their combined effect should still model y.
    # The normal equation solver with pseudo-inverse (pinv) should find a solution.
    # Let's set a simple y: y = X1 + X2 + const = X1 + 2*X1 + const = 3*X1 + const
    # Example: y = 3*X1 + 2
    y_np = 3 * X_pl.select("X1").to_series().to_numpy() + 2.0
    y_pl = pl.Series("y", y_np)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_pl, y_pl)

    assert model.coef_ is not None
    assert model.intercept_ is not None

    # We can't easily assert specific coefficients due to collinearity.
    # For example, X1_coef + 2*X2_coef should be approx 3.
    # And intercept should be approx 2.
    # The sum of coefficients might be one way if X1 and X2 were scaled, or check predictions.

    # Instead, we check if the predictions are close to the actual y values.
    # This demonstrates that the model can still make good predictions.
    predictions_pl = model.predict(X_pl)
    np.testing.assert_array_almost_equal(
        predictions_pl.to_numpy(), y_pl.to_numpy(), decimal=5
    )

    # Check one specific combination:
    # If solution is [c1, c2] for coefs, then c1*X1 + c2*X2 = (c1 + 2*c2)*X1 should equal 3*X1
    # So, c1 + 2*c2 should be approx 3.0
    # And intercept_ should be approx 2.0
    # Note: np.linalg.pinv will give one of the many possible solutions.
    # For X = [[1,2],[2,4]], y = [5,10] (y=x1+x2+c; y=3x1+c)
    # X_b = [[1,1,2],[1,2,4]], y=[5,10]
    # One solution from sklearn: coef_ = [3, 0], intercept_ = 2
    # Another: coef_ = [1,1], intercept_ = 2
    # Another: coef_ = [0, 1.5], intercept_ = 2
    # The solution from pinv can be sensitive.
    # print(f"Collinear Coefs: {model.coef_}, Intercept: {model.intercept_}")
    np.testing.assert_almost_equal(model.intercept_, 2.0, decimal=5)
    if model.coef_ is not None: # Should be set
        assert len(model.coef_) == 2
        # Check that X1*coef_[0] + X2*coef_[1] + intercept_ = y
        # Test the relationship: coef_[0] + 2*coef_[1] should be close to 3
        np.testing.assert_almost_equal(model.coef_[0] + 2*model.coef_[1], 3.0, decimal=5)


  def test_fit_output_type(self):
    """Tests that fit returns self."""
    X_pl = pl.DataFrame({"X1": [1.0, 2.0]})
    y_pl = pl.Series("y", [1.0, 2.0])
    model = LinearRegression()
    assert model.fit(X_pl, y_pl) is model

  def test_predict_output_type_and_name(self):
    """Tests predict output type and series name."""
    X_pl = pl.DataFrame({"X1": [1.0, 2.0, 3.0]})
    y_pl = pl.Series("y", [2.0, 4.0, 6.0])
    model = LinearRegression().fit(X_pl, y_pl)
    predictions = model.predict(X_pl)
    assert isinstance(predictions, pl.Series)
    assert predictions.name == "predictions"

  def test_input_X_polars_df_y_polars_series(self):
    """Tests fit with Polars DataFrame X and Polars Series y."""
    X_df = pl.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_series = pl.Series([1, 2, 3])
    model = LinearRegression()
    # Should fit without errors
    model.fit(X_df, y_series)
    assert model.coef_ is not None

  def test_input_X_1d_polars_df(self):
    """Tests predict with a 1D Polars DataFrame X (single feature)."""
    X_fit = pl.DataFrame({'feature1': [1, 2, 3]})
    y_fit = pl.Series([1, 2, 3])
    model = LinearRegression()
    model.fit(X_fit, y_fit)

    X_predict = pl.DataFrame({'feature1': [4, 5]})
    predictions = model.predict(X_predict)
    assert len(predictions) == 2
    assert isinstance(predictions, pl.Series)

  def test_coef_intercept_types_and_shapes(self):
    """Tests the types and shapes of coef_ and intercept_."""
    X_pl = pl.DataFrame({
        "X1": [1.0, 2.0, 3.0],
        "X2": [2.0, 3.0, 2.5],
    })
    y_pl = pl.Series("y", [1.0, 2.0, 3.0])

    # With intercept
    model_intercept = LinearRegression(fit_intercept=True)
    model_intercept.fit(X_pl, y_pl)
    assert isinstance(model_intercept.coef_, np.ndarray)
    assert model_intercept.coef_.ndim == 1
    assert model_intercept.coef_.shape[0] == X_pl.shape[1]
    assert isinstance(model_intercept.intercept_, float)

    # Without intercept
    model_no_intercept = LinearRegression(fit_intercept=False)
    model_no_intercept.fit(X_pl, y_pl)
    assert isinstance(model_no_intercept.coef_, np.ndarray)
    assert model_no_intercept.coef_.ndim == 1
    assert model_no_intercept.coef_.shape[0] == X_pl.shape[1]
    assert isinstance(model_no_intercept.intercept_, float) # Should be 0.0
    assert model_no_intercept.intercept_ == 0.0

    # Single feature
    X_single_feat = X_pl.select("X1")
    model_single_intercept = LinearRegression(fit_intercept=True)
    model_single_intercept.fit(X_single_feat, y_pl)
    assert isinstance(model_single_intercept.coef_, np.ndarray)
    assert model_single_intercept.coef_.ndim == 1
    assert model_single_intercept.coef_.shape[0] == X_single_feat.shape[1]
    assert isinstance(model_single_intercept.intercept_, float)

    model_single_no_intercept = LinearRegression(fit_intercept=False)
    model_single_no_intercept.fit(X_single_feat, y_pl)
    assert isinstance(model_single_no_intercept.coef_, np.ndarray)
    assert model_single_no_intercept.coef_.ndim == 1
    assert model_single_no_intercept.coef_.shape[0] == X_single_feat.shape[1]
    assert isinstance(model_single_no_intercept.intercept_, float)
    assert model_single_no_intercept.intercept_ == 0.0
