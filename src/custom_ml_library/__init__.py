"""Custom ML Library main package."""

from .classification import KNeighborsClassifier
from .cluster import DBSCAN  # Added DBSCAN as it is now moved.
from .preprocessing import MinMaxScaler, StandardScaler
from .regression import LinearRegression
from .utils import euclidean_distance

__all__ = [
    "KNeighborsClassifier",
    "LinearRegression",
    "MinMaxScaler",
    "StandardScaler",
    "DBSCAN",  # Added DBSCAN here as well.
    "euclidean_distance",
]
