from transformers.LengthTransformer import AverageWordLengthExtractor, TextLengthExtractor
import pandas as pd
from transformers.ColumnSelector import ColumnSelector
from transformers.PosCountTransformer import PosCountExtractor
from sklearn.pipeline import make_pipeline, make_union
from ml_utils import read_stanford_imdb_data, read_sentiment_datasets

def add_features(df):
    X = df.drop(['sentiment'], axis=1)

    t1 = ColumnSelector(cols=['review'])
    t2 = AverageWordLengthExtractor(result_col_name='avg_len', text_col_name='review')
    t3 = TextLengthExtractor(result_col_name='review_len', text_col_name='review')
    t4 = PosCountExtractor(pos_name='noun', result_col_name='noun_count', text_col_name='review')
    t5 = PosCountExtractor(pos_name='verb', result_col_name='verb_count', text_col_name='review')
    t6 = PosCountExtractor(pos_name='adj', result_col_name='adj_count', text_col_name='review')
    t7 = PosCountExtractor(pos_name='adv', result_col_name='adv_count', text_col_name='review')
    t8 = PosCountExtractor(pos_name='pron', result_col_name='pron_count', text_col_name='review')

    # I expect the make_union to produce 2 additional columns
    pipe = make_union(t1, t2, t3, t4, t5, t6, t7, t8, n_jobs=-1)

    n = pipe.transform(X)

    _transformed_df = pd.DataFrame(n, columns=['review', 'avg_len', 'review_len', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count'])
    return _transformed_df

if __name__ == '__main__':

    df = read_sentiment_datasets()
    print(f"imdb column names: {df.columns}")
    print(f"imdb shape: {df.shape}")

    print(df.head(20))
    print(df.tail(20))

    transformed_df = add_features(df)
    print('Transformed DataFrame')
    print(f"column names: {transformed_df.columns}")
    print(f"shape: {transformed_df.shape}")

    print(transformed_df.head(20))
    print(transformed_df.tail(20))



