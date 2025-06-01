"""Data preprocessing tools for the Custom ML Library.

This module provides a suite of tools for preparing data for machine learning models,
including feature scaling, normalization, encoding of categorical features,
and imputation of missing values.
"""
# This file makes the preprocessing directory a Python package.

from .scalers import MinMaxScaler, RobustScaler, StandardScaler
from .normalization import Normalizer
from .encoders import LabelEncoder, OneHotEncoder
from .imputation import SimpleImputer # Added SimpleImputer

__all__ = [
    "LabelEncoder",
    "MinMaxScaler",
    "Normalizer",
    "OneHotEncoder",
    "RobustScaler",
    "SimpleImputer", # Added SimpleImputer
    "StandardScaler",
]  # Kept alphabetical order
