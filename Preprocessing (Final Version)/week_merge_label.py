'''
Created in 2024
By Otto Gaulke
Description: consolidates weeks data files and creates non-adherence labels.
'''
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

df = pd.DataFrame()

# load all weeks and merge
for x in range(1, 11):
    print(f'Loading week {x}...')
    temp = pd.read_csv(f'weeks_{x}..csv',
                       low_memory=False)

    df = pd.concat([df, temp],
                   axis=0)

df.reset_index(drop=True,
               inplace=True)

# create new columns to hold adherence labels
# Non_adhere will flag for any non-adherence in the data
df['Non_adhere'] = np.nan
# Non_adhere_last4 will flag for non-adherence in the last 4 weeks (the
# prediction window)
df['Non_adhere_last4'] = np.nan

# iterate through the cas_nums columns
for index, val in df['cas_nums'].items():
    print('Processing adherence:' +
          str(round(round(index / df['cas_nums'].shape[0], 2) * 100, 0)) +
          '%')

    # check whether the flag for diabetes medication non-adherence is present
    if '172' in val.split(','):
        # if so, add a non-adherence flag
        df.at[index, 'Non_adhere'] = 1

# fill NaN values with 0
df.loc[pd.isna(df['Non_adhere']), 'Non_adhere'] = 0

print('Generating Labels...')

# if the member has a flag for non-adherence in the last 4 weeks then
# Non_adhere_last4 is 1
df.loc[(df['Non_adhere'] == 1) &
       (df['Weekcount'] > 48), 'Non_adhere_last4'] = 1
# fill NaN values with 0
df.loc[pd.isna(df['Non_adhere_last4']), 'Non_adhere_last4'] = 0

# convert to integer type
df['Non_adhere'] = df['Non_adhere'].astype(int)
df['Non_adhere_last4'] = df['Non_adhere_last4'].astype(int)

# now obtain the CLAB_ID of members that remained adherent for the first
# 48 weeks and became non-adherent in the last 4 weeks (flipped)

# create a new dataframe holding only the first 48 weeks of data
members = df[df['Weekcount'] < 49]
# group by CLAB_ID and sum Non_adhere labels (if a member became non-adherent
# in the first 48 weeks, their value will be greater than 0
members = members.groupby('CLAB_ID')['Non_adhere'].sum()
# drop members who became non-adherent at any point in the first 48 weeks
# save the adherent members' CLAB_ID
members = members[members == 0].index.tolist()
# create a new dataframe and isolate the data so that there only members
# who remained adherent in the first 48 weeks using the CLAB_ID's from above
adhere_48 = df[df['CLAB_ID'].isin(members)]
# accept only members who remained adherent in the first 48 weeks and
# became non-adherent in the last 4 weeks
flip = adhere_48[adhere_48['Non_adhere_last4'] == 1]
# save the CLAB_ID for these members
flip_members = flip['CLAB_ID'].tolist()

# create a dataframe to hold non-adherence labels
# group for a single row per member with their non-adherence counts
labels = df[['CLAB_ID',
             'Non_adhere',
             'Non_adhere_last4']] \
                .groupby('CLAB_ID')['Non_adhere',
                                    'Non_adhere_last4'] \
                .sum()

labels['CLAB_ID'] = labels.index

labels.reset_index(drop=True,
                   inplace=True)

labels = labels[['CLAB_ID',
                 'Non_adhere',
                 'Non_adhere_last4']]

# create non-adherence flip variable
labels['Non_adhere_flip'] = np.nan
# turn non-adherence counts into flags
labels.loc[labels['Non_adhere'] > 0, 'Non_adhere'] = 1
labels.loc[labels['Non_adhere_last4'] > 0, 'Non_adhere_last4'] = 1
# flag members who flipped in the last 4 weeks
labels.loc[labels['CLAB_ID'].isin(flip_members), 'Non_adhere_flip'] = 1
labels.loc[pd.isna(labels['Non_adhere_flip']), 'Non_adhere_flip'] = 0

# drop non-adherence columns from the consolidated weeks data
df.drop(columns=['Non_adhere', 'Non_adhere_last4'],
        inplace=True)

print('Saving Files...')

# save the consolidated weeks data
df.to_csv('weeks_full.csv',
          index=False)

# save the non-adherence labels
labels.to_csv('member_labels.csv',
              index=False)