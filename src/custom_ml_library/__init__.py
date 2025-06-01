"""Custom ML Library main package."""

from .classification import KNeighborsClassifier
from .cluster import DBSCAN
from .preprocessing import (  # Formatted for readability
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    RobustScaler,
    SimpleImputer,  # Added SimpleImputer
    StandardScaler,
)
from .regression import LinearRegression
from .utils import euclidean_distance

# Corrected __all__ list (alphabetical for easier maintenance):
__all__ = sorted([
    "KNeighborsClassifier",
    "LinearRegression",
    "LabelEncoder",
    "MinMaxScaler",
    "Normalizer",
    "OneHotEncoder",
    "RobustScaler",
    "SimpleImputer",  # Added SimpleImputer
    "StandardScaler",
    "DBSCAN",
    "euclidean_distance",
])
