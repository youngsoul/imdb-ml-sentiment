from sklearn.base import BaseEstimator, TransformerMixin

"""
Very simple transformer to select a few columns


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, make_pipeline, make_union


# Import libraries and download example data
from sklearn.preprocessing import StandardScaler, OneHotEncoder

dataset = pd.read_csv("./binary.csv")

# Define which columns should be encoded vs scaled
columns_to_encode = ['rank']
columns_to_scale  = ['gre', 'gpa']

# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        return df[self.key]

p = make_pipeline(
        make_union(
            make_pipeline(ItemSelector(key=columns_to_scale), scaler),
            make_pipeline(ItemSelector(key=columns_to_encode), ohe)
        )
    )

print(pd.DataFrame(p.fit_transform(dataset)).head())

"""

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        return df[self.key]
