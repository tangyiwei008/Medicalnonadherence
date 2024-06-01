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
         rx_merge_prep=False, weeks_merge_prep=False,
         w1=False, w2=False, w3=False, w4=False):
    # step the current working directory to the folder holding this script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # run dependent preprocessing scripts passed by function call
    if member_clean is True:
        print('Cleaning member data...')
        os.system('python CLAB_MEMBER_encoding.py')

    if rx_claims_merge is True:
        print('Merging RX and claims data...')
        os.system('python claims_rx_merge.py')

    if recode is True:
        print('Discretizing claims data...')
        os.system('python rxclaims_recoding.py')

    if rx_merge_prep is True:
        print('Running RX/claims merge prep...')
        os.system('python rxclaims_aggregation.py')

    if weeks_merge_prep is True:
        print('Running weeks merge prep...')
        os.system('python weeks_aggregation.py')

    print('Loading member data...')
    member = pd.read_csv('MEMBER_enc.csv',
                         low_memory=False)

    # accumulate list of weeks to be merged
    weeks_list = []

    if w1 is True:
        weeks_list.append('1')

    if w2 is True:
        weeks_list.append('2')

    if w3 is True:
        weeks_list.append('3')

    if w4 is True:
        weeks_list.append('4')

    # load aggregated rxclaims and weeks data and call merging function
    for w in weeks_list:
        print(f'Loading rx/claims {w} week data...')
        df_rxclaims = pd.read_csv(f'rxclaims_week_{w}.csv',
                                  low_memory=False)

        print(f'Loading weeks {w} week data...')
        df_week = pd.read_csv(f'ipro_week_{w}.csv',
                               low_memory=False)

        if w == '1':
            merger(rx=df_rxclaims,
                   week=df_week,
                   member=member,
                   w1=True)
        elif w == '2':
            merger(rx=df_rxclaims,
                   week=df_week,
                   member=member,
                   w2=True)
        elif w == '3':
            merger(rx=df_rxclaims,
                   week=df_week,
                   member=member,
                   w3=True)
        elif w == '4':
            merger(rx=df_rxclaims,
                   week=df_week,
                   member=member,
                   w4=True)

        del df_rxclaims, df_week


# merge claims and weeks data
def merger(rx=None, week=None,
           member=None,
           w1=False, w2=False, w3=False, w4=False):
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

    # call saving function
    if (w1 is True and
        w2 is False and
        w3 is False and
        w4 is False):

        print('Saving file...')
        save(df,
             w1=True)

        del df
    elif (w1 is False and
          w2 is True and
          w3 is False and
          w4 is False):

        print('Saving file...')
        save(df,
             w2=True)

        del df
    elif (w1 is False and
          w2 is False and
          w3 is True and
          w4 is False):

        print('Saving file...')
        save(df,
             w3=True)

        del df
    elif (w1 is False and
          w2 is False and
          w3 is False and
          w4 is True):

        print('Saving file...')
        save(df,
             w4=True)

        del df


# save merged data
def save(df,
         w1=False, w2=False, w3=False, w4=False):
    # load in dataframe of common member indexes
    df_member_temp = pd.read_csv('members_index.csv',
                                 low_memory=False)

    # restrict members in dataframe using indexes
    df = df[df['CLAB_ID'].isin(df_member_temp['CLAB_ID'])]

    # save files
    if (w1 is True and
        w2 is False and
        w3 is False and
        w4 is False):

        df.to_csv('merged_model_1week.csv',
                  index=False)

    elif (w1 is False and
          w2 is True and
          w3 is False and
          w4 is False):

        df.to_csv('merged_model_2week.csv',
                  index=False)

    elif (w1 is False and
          w2 is False and
          w3 is True and
          w4 is False):

        df.to_csv('merged_model_3week.csv',
                  index=False)

    elif (w1 is False and
          w2 is False and
          w3 is False and
          w4 is True):

        df.to_csv('merged_model_4week.csv',
                  index=False)


main(member_clean=False,
     rx_claims_merge=False,
     recode=False,
     rx_merge_prep=False,
     weeks_merge_prep=False,
     w1=True,
     w2=True,
     w3=True,
     w4=True)
