import pandas as pd

root_dataset_dir = '/Users/patrickryan/Development/python/mygithub/imdb-ml-sentiment/data'


def read_stanford_imdb_data():
    base_path = '/stanford_imdb'

    df = pd.read_csv(f'{root_dataset_dir}/{base_path}/imdb_df.csv.gzip', compression='gzip')

    return df


def read_sentiment_datasets() -> pd.DataFrame:
    base_path = '/sentiment_labelled_sentences'

    filepath_dict = {
        'yelp': f'{root_dataset_dir}/{base_path}/yelp_labelled.txt',
        'amazon': f'{root_dataset_dir}/{base_path}/amazon_cells_labelled.txt',
        'imdb': f'{root_dataset_dir}/{base_path}/imdb_labelled.txt'
    }

    df_list = []

    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['review', 'sentiment'], sep='\t')
        df['source'] = source
        df_list.append(df)

    df = pd.concat(df_list)

    return df
