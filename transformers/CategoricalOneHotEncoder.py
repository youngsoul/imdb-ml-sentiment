from pandas import Categorical, get_dummies
from sklearn.base import TransformerMixin, BaseEstimator

"""
https://masongallo.github.io/machine/learning,/python/2017/10/07/machine-learning-pipelines.html
"""

class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    """One hot encoder for all categorical features"""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        cats = {}
        for column in self.attribute_names:
            cats[column] = X[column].unique().tolist()
        self.categoricals = cats
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for column in self.attribute_names:
            df[column] = Categorical(df[column], categories=self.categoricals[column])
        new_df = get_dummies(df, drop_first=True)
        # in case we need them later
        self.columns = new_df.columns
        return new_df