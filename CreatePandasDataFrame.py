import pandas as pd
import os
import pyprind


base_path = '/Volumes/MacBackup/aclImdb'

labels = {'pos': 1, 'neg': 0}

df = pd.DataFrame(columns=['review', 'sentiment'])
pbar = pyprind.ProgBar(50000)

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(base_path, s, l) # base/test/pos, base/test/neg, base/train/pos, base/train/neg
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()

            # print(labels[l], txt )
            df = df.append({'review': txt, 'sentiment': labels[l]}, ignore_index=True)
            pbar.update()


print(df)

df.to_csv(os.path.join(base_path, 'imdb_df.csv.gzip'), compression="gzip")
