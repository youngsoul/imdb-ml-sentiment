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

    # I expect the make_union to produce 2 additional columns
    pipe = make_union(t1, t2, ColumnSelector(cols=['review']), PosCountExtractor(pos_name='noun', result_col_name='noun_count', text_col_name='review'))


    n = pipe.transform(X)

    print(n.shape)
    print(type(n))  # numpy.ndarray
