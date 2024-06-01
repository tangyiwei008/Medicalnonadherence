'''
Created on Feb 11 2024
By Otto Gaulke
Description: Connects all preprocessing files and merges the final data.
All dependencies (.csv's and .py's) should be contained
in the same folder as this master script. Allows the user to select which
preprocessing scripts to run as needed. Merges the RX/claims,
member, and weekly data into final data .csv's.
'''
import pandas as pd
import os

# run necessary preprocessing scripts (if specified) and start merge function
def main(member_clean=False, rx_claims_merge=False, weeks_consolidate=False,
         recode=False, rx_merge_prep=False, weeks_merge_prep=False):
    # step the current working directory to the folder holding this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # run dependent preprocessing scripts passed by function call
    if member_clean is True:
        print('Cleaning member data...')
        os.system('python CLAB_MEMBER_encoding_prod.py')

    if rx_claims_merge is True:
        print('Merging RX and claims data...')
        os.system('python claims_rx_merge_prod.py')

    if weeks_consolidate is True:
        print('Merging Weeks data...')
        os.system('python week_merge_label_prod.py')

    if recode is True:
        print('Discretizing claims data...')
        os.system('python rxclaims_recoding_prod.py')

    if rx_merge_prep is True:
        print('Running RX/claims merge prep...')
        os.system('python rxclaims_aggregation_prod.py')

    if weeks_merge_prep is True:
        print('Running weeks merge prep...')
        os.system('python weeks_aggregation_prod.py')

    print('Loading member data...')
    member = pd.read_csv('MEMBER_enc.csv',
                         low_memory=False)

    # load aggregated rxclaims and weeks data and call merging function
    print(f'Loading rx/claims data...')
    df_rxclaims = pd.read_csv(f'rxclaims_final.csv',
                              low_memory=False)

    print(f'Loading weeks data...')
    df_week = pd.read_csv(f'weeks_final.csv',
                           low_memory=False)

    member['CLAB_ID'] = member['CLAB_ID'].astype(int)
    df_rxclaims['CLAB_ID'] = df_rxclaims['CLAB_ID'].astype(int)
    df_week['CLAB_ID'] = df_week['CLAB_ID'].astype(int)

    merger(rx=df_rxclaims,
           week=df_week,
           member=member)


# merge claims and weeks data
def merger(rx=None, week=None, member=None):
    # merge member and week data
    print('Merging week data...')
    df = pd.merge(member,
                  week,
                  how='inner',
                  on=['CLAB_ID'])

    print('Merging rxclaims data...')
    # merge rxclaims data
    df = pd.merge(df,
                  rx,
                  how='inner',
                  on=['CLAB_ID'])

    print('Saving file...')
    save(df)


# save merged data
def save(df):

    df.to_csv('merged_final.csv',
              index=False)

main(member_clean=False,
     rx_claims_merge=False,
     weeks_consolidate=False,
     recode=False,
     rx_merge_prep=False,
     weeks_merge_prep=False)
