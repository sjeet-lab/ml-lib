"""Linear Regression model."""

import numpy as np
import polars as pl

class LinearRegression:
  """Linear regression model using Ordinary Least Squares.

  Attributes:
    fit_intercept: bool, whether to calculate the intercept for this model.
    coef_: np.ndarray, estimated coefficients for the linear regression problem.
    intercept_: float, independent term in the linear model. Set to 0.0 if
      `fit_intercept = False`.
  """

  def __init__(self, fit_intercept: bool = True):
    """Initializes LinearRegression.

    Args:
      fit_intercept: Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations (i.e., data is
        expected to be centered).
    """
    self.fit_intercept = fit_intercept
    self.coef_ = None
    self.intercept_ = None if fit_intercept else 0.0

  def fit(self, X: pl.DataFrame, y: pl.Series):
    """Fits the linear model.

    Calculates the coefficients and intercept (if applicable) using the
    normal equation.

    Args:
      X: Polars DataFrame of shape (n_samples, n_features), training data.
      y: Polars Series of shape (n_samples,), target values.

    Returns:
      self: Fitted estimator.
    """
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    if X_np.ndim == 1:
      X_np = X_np.reshape(-1, 1)

    if self.fit_intercept:
      X_np = np.concatenate([np.ones((X_np.shape[0], 1)), X_np], axis=1)

    # Normal equation: beta = (X^T * X)^(-1) * X^T * y
    try:
      # Adding a small epsilon to the diagonal for numerical stability
      # especially if XTX is singular or near singular.
      xtx = X_np.T @ X_np
      # Check if xtx is scalar (e.g. one feature, one sample with intercept)
      if xtx.ndim == 0: # Should not happen with practical inputs
          xtx_inv = 1.0 / xtx if xtx !=0 else np.array([[0.0]]) # Avoid division by zero
      elif xtx.shape == (1,1): # Single feature with intercept, or single feature no intercept
          xtx_inv = np.array([[1.0 / xtx[0,0]]]) if xtx[0,0] != 0 else np.array([[0.0]])
      else:
          # Using pseudo-inverse for robustness
          xtx_inv = np.linalg.pinv(xtx)

      beta = xtx_inv @ X_np.T @ y_np
    except np.linalg.LinAlgError:
      # Fallback or error reporting if pinv also fails, though pinv is robust.
      raise ValueError(
          "Failed to compute coefficients. This may be due to multicollinearity"
          " or other numerical issues."
      )


    if self.fit_intercept:
      self.intercept_ = beta[0]
      self.coef_ = beta[1:]
    else:
      self.coef_ = beta
      self.intercept_ = 0.0 # Ensure intercept is float

    # Ensure coef_ is always 1D array for consistency, even with one feature
    if self.coef_ is not None and self.coef_.ndim == 0:
        self.coef_ = np.array([self.coef_])
    elif self.coef_ is not None and self.coef_.ndim > 1 : # Should be (n_features,)
        self.coef_ = self.coef_.flatten()


    return self

  def predict(self, X: pl.DataFrame) -> pl.Series:
    """Predicts using the linear model.

    Args:
      X: Polars DataFrame of shape (n_samples, n_features), samples.

    Returns:
      Polars Series of shape (n_samples,), predicted values.

    Raises:
      RuntimeError: If the model has not been fitted yet.
    """
    if self.coef_ is None:
      raise RuntimeError(
          "This LinearRegression instance is not fitted yet. Call 'fit' with "
          "appropriate arguments before using this estimator."
      )

    X_np = X.to_numpy()
    if X_np.ndim == 1:
      X_np = X_np.reshape(-1, 1)

    # y_pred = X * self.coef_ + self.intercept_
    # Ensure coef_ is correctly shaped for dot product if X_np has multiple features
    # and coef_ is 1D array.
    # np.dot can handle (N, M) @ (M,) -> (N,)
    predictions_np = np.dot(X_np, self.coef_) + self.intercept_

    return pl.Series(values=predictions_np, name="predictions")
