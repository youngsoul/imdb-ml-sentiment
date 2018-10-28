from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def get_feature_names(self):
        return ['avg_word_len']

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        result_series = X['review'].apply(self.average_word_length)
        return result_series.to_frame(name=self.get_feature_names()[0])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def get_feature_names(self):
        return ['avg_len']

    def text_length(self, sample_text):
        """Helper code to compute average word length of a name"""
        return len(sample_text)

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        result_series = X['review'].apply(self.text_length)
        return result_series.to_frame(name=self.get_feature_names()[0])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


if __name__ == '__main__':
    from sklearn.pipeline import make_pipeline, make_union

    base_path = '../data/stanford_imdb'

    df = pd.read_csv(f'{base_path}/imdb_df.csv.gzip', compression='gzip')
    X = df.drop(['sentiment'], axis=1)
    

    t1 = AverageWordLengthExtractor()
    t2 = TextLengthExtractor()

    # I expect the make_union to produce 2 additional columns
    pipe = make_union(t1,t2)

    n = pipe.transform(X)

    print(n.shape)
    print(type(n)) #numpy.ndarray
