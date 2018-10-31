from transformers.LengthTransformer import AverageWordLengthExtractor, TextLengthExtractor
import pandas as pd
from transformers.ColumnSelector import ColumnSelector
from transformers.PosCountTransformer import PosCountExtractor

if __name__ == '__main__':
    from sklearn.pipeline import make_pipeline, make_union

    base_path = '../data/stanford_imdb'

    df = pd.read_csv(f'{base_path}/imdb_df.csv.gzip', compression='gzip')
    X = df.drop(['sentiment'], axis=1)

    t1 = AverageWordLengthExtractor(result_col_name='avg_len', text_col_name='review')
    t2 = TextLengthExtractor(result_col_name='review_len', text_col_name='review')
    t3 = ColumnSelector(cols=['review'])
    t4 = PosCountExtractor(pos_name='noun', result_col_name='noun_count', text_col_name='review')
    t5 = PosCountExtractor(pos_name='verb', result_col_name='verb_count', text_col_name='review')
    t6 = PosCountExtractor(pos_name='adj', result_col_name='adj_count', text_col_name='review')
    t7 = PosCountExtractor(pos_name='adv', result_col_name='adv_count', text_col_name='review')
    t8 = PosCountExtractor(pos_name='pron', result_col_name='pron_count', text_col_name='review')

    # I expect the make_union to produce 2 additional columns
    pipe = make_union(t1, t2, t3, t4, t5, t6, t7, t8, n_jobs=-1)


    n = pipe.transform(X)

    print(n.shape)
    print(type(n))  # numpy.ndarray
    transformed_df = pd.DataFrame(n)

    print(transformed_df.head())
