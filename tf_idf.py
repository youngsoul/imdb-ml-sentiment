import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

base_path = '/Volumes/MacBackup/aclImdb'

df = pd.read_csv(f'{base_path}/imdb_df.csv.gzip', compression='gzip')

print(df)

