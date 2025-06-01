# Custom ML Library

A Python library for custom implementations of common machine learning algorithms, built with a focus on clarity and using Polars for data handling where appropriate.

## Overview

This library provides a growing collection of machine learning tools, allowing users to understand the inner workings of algorithms and to have fine-grained control over their execution. It's designed for both educational purposes and practical application where custom or specific algorithm versions are needed.

## Module Structure

The library is organized into the following modules:

*   **`classification`**: Contains algorithms for classification tasks.
    *   `KNeighborsClassifier`: K-Nearest Neighbors classifier.
*   **`regression`**: Implements algorithms for regression problems.
    *   `LinearRegression`: Ordinary Least Squares Linear Regression.
*   **`preprocessing`**: Includes tools for data preprocessing.
    *   `StandardScaler`: Standardizes features by removing the mean and scaling to unit variance.
    *   `MinMaxScaler`: Scales features to a given range, typically [0, 1].
*   **`cluster`**: Houses clustering algorithms.
    *   `dbscan.py`: (Contains an existing DBSCAN implementation - details to be added).
*   **`utils`**: Provides utility functions supporting various algorithms.
    *   `distances.py`: Includes functions like `euclidean_distance`.

## Basic Usage Example

Here's a quick example of how to use a scaler:

```python
import polars as pl
from preprocessing import StandardScaler # Assuming __init__.py setup for direct import

# Sample data in a Polars DataFrame
data = pl.DataFrame({
    "feature1": [10.0, 20.0, 30.0],
    "feature2": [1.0, 2.0, 3.0]
})

# Initialize and use the scaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print(scaled_data)
```

To use a model:

```python
from classification import KNeighborsClassifier
from regression import LinearRegression # Example for regression
# ... prepare your X_train_df, y_train_series, X_test_df ...

# For classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_df, y_train_series)
predictions = knn.predict(X_test_df)

print(predictions)
```

## Project Status

This project is currently in the initial development phase. More algorithms, features, comprehensive tests, and detailed documentation are planned for future updates. Contributions and feedback are welcome!

## License

(To be added - e.g., MIT License)
```

Use code with caution.
