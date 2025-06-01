# Custom ML Library

A Python library for custom implementations of common machine learning algorithms, built with a focus on clarity and using Polars for data handling where appropriate.

## Overview

This library provides a growing collection of machine learning tools, allowing users to understand the inner workings of algorithms and to have fine-grained control over their execution. It's designed for both educational purposes and practical application where custom or specific algorithm versions are needed.

## Module Structure

The library is organized into the following modules (found under `src/custom_ml_library`):

*   **`classification`**: Contains algorithms for classification tasks.
    *   `KNeighborsClassifier`: K-Nearest Neighbors classifier.
*   **`regression`**: Implements algorithms for regression problems.
    *   `LinearRegression`: Ordinary Least Squares Linear Regression.
*   **`preprocessing`**: Includes tools for data preprocessing.
    *   `StandardScaler`: Standardizes features by removing the mean and scaling to unit variance.
    *   `MinMaxScaler`: Scales features to a given range (e.g., 0-1).
    *   `RobustScaler`: Scales features using statistics robust to outliers (median and Interquartile Range).
    *   `Normalizer`: Normalizes samples individually to unit norm (L1 or L2).
    *   `LabelEncoder`: Encodes target labels with values between 0 and n_classes-1.
    *   `OneHotEncoder`: Encodes categorical integer features as a one-hot numeric array.
    *   `SimpleImputer`: Imputes missing values using common strategies (mean, median, most frequent, constant).
*   **`cluster`**: Houses clustering algorithms.
    *   `DBSCAN`: Density-Based Spatial Clustering of Applications with Noise.
*   **`utils`**: Provides utility functions supporting various algorithms.
    *   `euclidean_distance`: Calculates Euclidean distance.

## Basic Usage Example

Here's a quick example of how to use some preprocessing tools:

```python
import polars as pl
import numpy as np
from custom_ml_library.preprocessing import StandardScaler, SimpleImputer, LabelEncoder

# Sample data with missing values and categorical labels
data = pl.DataFrame({
    "feature1": [10.0, 20.0, None, 40.0, 10.0],
    "feature2": [1.0, None, 3.0, 4.0, 1.0],
    "labels": ["cat", "dog", "mouse", "cat", "dog"]
})

# Impute missing values
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data.select(["feature1", "feature2"]))

# Scale numerical features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)
print("Scaled Data:")
print(data_scaled)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data.select("labels").to_series())
print("\nEncoded Labels:")
print(encoded_labels)
print("Classes:", label_encoder.classes_)
```

To use a model:

```python
from custom_ml_library.classification import KNeighborsClassifier

# Sample data (replace with your actual data)
X_train_df = pl.DataFrame({"feature1": [1.0, 2.0, 6.0, 7.0], "feature2": [2.0, 3.0, 7.0, 8.0]})
y_train_series = pl.Series("labels", [0, 0, 1, 1]) # Assuming labels are already encoded for KNN
X_test_df = pl.DataFrame({"feature1": [3.0], "feature2": [4.0]})


# For classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_df, y_train_series)
predictions = knn.predict(X_test_df)

print("\nKNN Predictions:")
print(predictions)
```

## Project Status

This project is currently in the initial development phase. More algorithms, features, comprehensive tests, and detailed documentation are planned for future updates. Contributions and feedback are welcome!

## Development Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.
Pre-commit hooks are configured to ensure code quality and consistency.

1.  **Install Poetry:**
    Follow the official [installation guide](https://python-poetry.org/docs/#installation).

2.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd <repository-name>   # Replace <repository-name> with the cloned directory name
    ```

3.  **Install dependencies:**
    This will create a virtual environment if one doesn't exist and install all project dependencies, including development tools.
    The command `poetry install` by default installs all dependencies including those specified in `[tool.poetry.group.dev.dependencies]`.
    ```bash
    poetry install
    ```

4.  **Activate the virtual environment:**
    Poetry creates its virtual environments in a central cache or within a `.venv` directory in the project (if `poetry config virtualenvs.in-project true` has been set).
    To run commands within the project's environment, it's often easiest to use `poetry run <command>`.
    For example, to run tests:
    ```bash
    poetry run pytest
    ```
    If you prefer to activate the shell, first find the environment path:
    ```bash
    poetry env info --path
    ```
    Then activate it (e.g., `source <path-to-venv>/bin/activate` on Linux/macOS or `<path-to-venv>\Scripts\activate.bat` on Windows).

5.  **Set up pre-commit hooks:**
    The pre-commit hooks are defined in `.pre-commit-config.yaml`. To install them into your local git repository:
    ```bash
    poetry run pre-commit install
    ```
    This will ensure that checks are run automatically before each commit. You can also run them manually on all files:
    ```bash
    poetry run pre-commit run --all-files
    ```

To run tests:
```bash
poetry run pytest
```

## License

(To be added - e.g., MIT License)
```

Use code with caution.
