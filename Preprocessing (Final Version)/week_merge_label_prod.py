import pandas as pd
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

df = pd.DataFrame()

for x in range(1, 11):
    print(f'Loading week {x}...')
    temp = pd.read_csv(f'weeks_{x}..csv',
                       low_memory=False)

    df = pd.concat([df, temp],
                   axis=0)

df.reset_index(drop=True,
               inplace=True)

print('Saving Files...')

df.to_csv('weeks_full.csv',
          index=False)