'''
Created on Feb 11 2024
By Otto Gaulke
Description: Connects all preprocessing files and merges the final data
(1 week forecasting and 4 week forecasting). All dependencies
(.csv's and .py's) should be contained
in the same folder as this master script. Allows the user to select which
preprocessing scripts to run as needed. Merges the RX/claims,
member, and weekly data into final data .csv's.
'''
import pandas as pd
import os


# run necessary preprocessing scripts (if specified) and start merge function
def main(member_clean=False, rx_claims_merge=False, recode=False,
         rx_merge_prep=False, weeks_merge_prep=False):
    # step the current working directory to the folder holding this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # run dependent preprocessing scripts passed by function call
    if member_clean is True:
        print('Cleaning member data...')
        os.system('python CLAB_MEMBER_encoding_new_data.py')

    if rx_claims_merge is True:
        print('Merging RX and claims data...')
        os.system('python claims_rx_merge_new_data.py')

    if recode is True:
        print('Discretizing claims data...')
        os.system('python rxclaims_recoding_new_data.py')

    if rx_merge_prep is True:
        print('Running RX/claims merge prep...')
        os.system('python merge_rx_claims_new_data.py')

    if weeks_merge_prep is True:
        print('Running weeks merge prep...')
        os.system('python merge_weeks_new_data.py')

    print('Loading member data...')
    member = pd.read_csv('MEMBER_enc_new.csv',
                         low_memory=False)

    # load aggregated rxclaims and weeks data and call merging function
    print(f'Loading rx/claims data...')
    df_rxclaims = pd.read_csv(f'rxclaims_new.csv',
                              low_memory=False)

    print(f'Loading weeks data...')
    df_week = pd.read_csv(f'ipro_new.csv',
                           low_memory=False)

    merger(rx=df_rxclaims,
           week=df_week,
           member=member)

    del df_rxclaims, df_week


# merge claims and weeks data
def merger(rx=None, week=None,
           member=None):
    # merge member and week data
    print('Merging week data...')
    df = pd.merge(member,
                  week,
                  how='inner',
                  on=['CLAB_ID'])

    del member, week

    # merge rxclaims data
    df = pd.merge(df,
                  rx,
                  how='inner',
                  on=['CLAB_ID'])

    del rx

    print('Saving file...')
    save(df)

    del df


# save merged data
def save(df):

    df.to_csv('merged_new_data.csv',
              index=False)


main(member_clean=False,
     rx_claims_merge=False,
     recode=False,
     rx_merge_prep=False,
     weeks_merge_prep=False)
