from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self, result_col_name='noun_count', text_col_name=None):
        self.text_col_name = text_col_name
        self.result_col_name = result_col_name

    def get_feature_names(self):
        return [self.result_col_name]

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        X[self.result_col_name] = X[self.text_col_name].apply(self.average_word_length)
        return X[self.result_col_name].to_frame(name=self.get_feature_names()[0])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self, result_col_name='noun_count', text_col_name=None):
        self.text_col_name = text_col_name
        self.result_col_name = result_col_name

    def get_feature_names(self):
        return [self.result_col_name]

    def text_length(self, sample_text):
        """Helper code to compute average word length of a name"""
        return len(sample_text)

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        X[self.result_col_name] = X[self.text_col_name].apply(self.text_length)
        return X[self.result_col_name].to_frame(name=self.get_feature_names()[0])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


