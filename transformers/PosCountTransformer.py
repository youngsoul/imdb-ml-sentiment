from sklearn.base import BaseEstimator, TransformerMixin
import textblob


class PosCountExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, counts the number of time the particular POS is found"""

    def __init__(self, pos_name='noun', result_col_name='noun_count', text_col_name=None):
        self.pos_name = pos_name
        self.result_col_name = result_col_name
        self.text_col_name = text_col_name
        self.pos_family = {
            'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
            'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
            'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adj': ['JJ', 'JJR', 'JJS'],
            'adv': ['RB', 'RBR', 'RBS', 'WRB']
        }

    def get_feature_names(self):
        return [self.result_col_name]

    # function to check and get the part of speech tag count of a words in a given sentence
    def check_pos_tag(self, x):
        cnt = 0
        flag = self.pos_name
        try:
            text_info = textblob.TextBlob(x)
            for tup in text_info.tags:
                ppo = list(tup)[1]
                if ppo in self.pos_family[flag]:
                    cnt += 1
        except:
            pass
        return cnt

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        X[self.result_col_name] = X[self.text_col_name].apply(self.check_pos_tag)
        return X[self.result_col_name].to_frame(name=self.get_feature_names()[0])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
