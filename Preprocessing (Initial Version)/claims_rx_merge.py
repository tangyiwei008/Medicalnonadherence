'''
Created on Feb 5 2024
By Otto Gaulke
Description: Removes original RX claims from the claims data, replaces with
new RX claims data, and saves the resulting dataframe as a single .csv file.
'''
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

print('Processing File: claims_1..csv')

# load in the first claims file
claims = pd.read_csv('claim_1..csv',
                     low_memory=False)

# load in the first rxclaims file
rxclaims = pd.read_csv('rxclaims_1..csv',
                       encoding='ANSI',
                       low_memory=False)

# create a list of all columns in rxclaims
cols_ord = rxclaims.columns.tolist()

# create a list of rxclaims data columns not already in original claims data
cols_new = [cols_ord[3]] + [cols_ord[7]] + cols_ord[21:77]

del rxclaims

# drop all RX claims from the first claims file
# will be replace by new rxclaims data
claims = claims[claims['UTILGRP_CD'] != 'RX']

# iteratively add rxclaims columns not already in claims data to
# claims dataframe and fill with NaN
for el in cols_new:
    claims[el] = np.nan

# reorder the columns to match the order in rxclaims data
claims = claims[cols_ord].copy()

file = 'claim_'

# repeat process for each claims file and combine into the claims dataframe
for x in range(2, 17):
    file_name = file + str(x) + '..csv'

    print('Processing File:', file_name)

    temp = pd.read_csv(file_name,
                       low_memory=False)

    temp = temp[temp['UTILGRP_CD'] != 'RX']

    for el in cols_new:
        temp[el] = np.nan

    temp = temp[cols_ord].copy()

    # add new processed claims file to the claims dataframe
    claims = pd.concat([claims, temp], axis=0)

file = 'rxclaims_'

# iterate through RX files and add to claims dataframe
for x in range(1, 21):
    file_name = file + str(x) + '..csv'

    print('Processing File:', file_name)

    temp = pd.read_csv(file_name,
                       encoding='ANSI',
                       low_memory=False)

    # add new rxclaims files to the claims dataframe
    claims = pd.concat([claims, temp], axis=0)

print('Saving file...')

# save the final processed file
claims.to_csv('rxclaims_full.csv',
              encoding='UTF-8',
              index=False)
