import pandas as pd
import os
import pyprind


base_path = '/Volumes/MacBackup/aclImdb'

labels = {'pos': 1, 'neg': 0}

df = pd.read_csv(f'{base_path}/imdb_df.csv.gzip', compression='gzip')

print(df)
