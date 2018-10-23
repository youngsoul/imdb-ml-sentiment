import pandas as pd
import re
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
from sklearn.svm import SVC
from TextNormalizer import TextNormalizer

base_path = '/Volumes/MacBackup/aclImdb'

english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedTfIdfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfIdfVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

def read_df():
    df = pd.read_csv(f'{base_path}/imdb_df.csv.gzip', compression='gzip')
    return df

def remove_html_lower(text):
    text = re.sub('<[^>]*', '', text)

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

    return text

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, english_stemmer)
    return stems


def do_gridsearch():
    """
    Pipeline(memory=None,
     steps=[('tfidfvectorizer', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.float64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=0.8, max_features=None, min_df=2,
        ngram_range=(1, 4), norm='l2', preprocessor=None, smooth...penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False))])
0.9034285714285715
{'logisticregression__C': 100, 'tfidfvectorizer__min_df': 2, 'tfidfvectorizer__ngram_range': (1, 4)}
"""
    df = read_df()
    X = df['review']
    y = df['sentiment']
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=222 )
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)

    stem_pipeline = make_pipeline(TextNormalizer(), tfidf, LogisticRegression())  # LogisticRegression(C=100) or MultinomialNB()
    param_grid = {'logisticregression__C': [10, 50, 100],
                  'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
                  'tfidfvectorizer__min_df': [2, 3, 4, 5]}
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2)
    grid = GridSearchCV(stem_pipeline, param_grid, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)

def do_crossval():
    """
        # MultinomialNB = 0.878
    # LogisticRegression(C=100) = 0.9029
    # SVC(kernel='linear', C=1) = DNF
    # LogisticRegression(C=100) w/ Stemming = 0.89
    # LogisticRegression(C=100) w/o TextNormalizer 0.90

    :return:
    """
    df = read_df()
    # X = df['review'].apply(remove_html_lower)

    X = df['review']
    y = df['sentiment']
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=222 )

    tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.8, ngram_range=(1,4))
    stem_pipeline = make_pipeline(TextNormalizer(), tfidf, LogisticRegression(C=100))
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2)

    scores = cross_val_score(stem_pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    print(scores, scores.mean())

if __name__ == '__main__':
    #do_gridsearch()

    do_crossval()

