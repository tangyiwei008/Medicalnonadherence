"""
Created on Feb 10 2024
By Spencer Stromback
Description: Aggregates Rx_claims data into a single row per member.
This is to be combined with claims data (after similar compression)
This program prepares the data for the 4-week prediction model.
"""

import pandas as pd
import numpy as np
import os
from functools import reduce

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Setting random seed for reproducibility
np.random.seed(42)

print('Load')
df = pd.read_csv('rxclaims_recoded.csv',
                 encoding='unicode_escape',
                 low_memory=False)

columns_to_drop = [
    'line_id', 'RX_line_id', 'COPAYDIFF_AMT', 'COPAYPEN_AMT',
    'PRICEQTY_NBR', 'SUPPLQTY_NBR', 'ADMINRTE_DS', 'STRENGTH_DS',
    'DOSEFORM_DS', 'DRUG_NM', 'GNRC_NM', 'GNRC_CD_NBR', 'MAINT_DS',
    'MLTSGL_DS', 'DAW_DS', 'GNRCBRND_DS', 'GPI_CD', 'GPI2_DS', 'GPI4_DS',
    'LABEL_NM', 'OTC_DS', 'COMPOUND_DS', 'NDC_KEY', 'PRICESCHED_TXT',
    'RXPROVQLF_CD', 'RXPROVQLF_DS', 'RXPROV_ID', 'REFILL_DS', 'DRUGSTAT_DS'
]

print('Preproc')

# Drop columns
df.drop(columns=columns_to_drop,
        inplace=True)

# Add change in dc column
# Step 1: Sort the DataFrame
df = df.sort_values(by=['CLAB_ID', 'GPI14_DS', 'days_from_index'])
# Step 2: Calculate the days since last fill
# calculate the difference in `days_from_index` between the rows within
# these groups.
df['days_since_last_fill'] = df \
    .groupby(['CLAB_ID','GPI14_DS'])['days_from_index'].diff()
# Step 3: Subtract this value from `DAYSSUPP_NBR` of the last time the
# drug was dispensed
# For the first fill, the difference will be NaN, so we keep it as is.
# For subsequent fills,
# we calculate the new value.
df['change_in_dc'] = df['DAYSSUPP_NBR'] - df['days_since_last_fill']
# Replace NaN values with 0 for the first fill
df['change_in_dc'].fillna(0, inplace=True)

# Combine ICD_10_DIAGLN_CD columns
df['combined_diagnoses'] = (df[['ICD10_DIAGLN_CD', 'DIAG1_CD', 'DIAG2_CD',
                                'DIAG3_CD', 'DIAG4_CD', 'DIAG5_CD']]
                            .apply(lambda row: list(set(filter(pd.notna,
                                                               row))),
                                   axis=1))
df.drop(['ICD10_DIAGLN_CD', 'DIAG1_CD',
         'DIAG2_CD', 'DIAG3_CD', 'DIAG4_CD', 'DIAG5_CD'], axis=1, inplace=True)

# df.sort_values(by=['CLAB_ID', 'days_from_index'],
# ascending=[True, False], inplace=True)
df.sort_values(by=['CLAB_ID', 'days_from_index'],
               ascending=[True, True],
               inplace=True)

df['Week'] = df['days_from_index'] // 7 + 1

df = df[df['Week'] < 49]

# create days since last claim variable
temp = 336 - df.groupby('CLAB_ID')['days_from_index'].max()

# Set Week_non_adhere to 49, remove later
df['Week_non_adhere'] = 49

# Reorder column names
cols = df.columns.tolist()
cols = cols[:2] + cols[-3:] + cols[2:-3]
df = df[cols]


# Step 1 & 2: Isolate the data for each member for
# 52 weeks prior to their Week_non_adhere
def filter_data_per_member(df):
    # This function will iterate over each member and filter their data
    filtered_dfs = []  # Store each member's filtered DataFrame

    for clab_id in df['CLAB_ID'].unique():
        member_df = df[df['CLAB_ID'] == clab_id]  # Data for the current member
        week_non_adhere = member_df['Week_non_adhere'].dropna().unique()

        # If member has a Week_non_adhere value
        if len(week_non_adhere) > 0:
            week_non_adhere = week_non_adhere[0]
            # Calculate the week range
            week_start = 1
            week_end = 48
            # Filter the member's data for the
            # 52 weeks prior to Week_non_adhere
            member_filtered = member_df[(member_df['Week'] >= week_start) &
                                        (member_df['Week'] <= week_end)]
            filtered_dfs.append(member_filtered)

    return pd.concat(filtered_dfs)  # Combine all filtered DataFrames into one

print('Filter')
# Apply the filtering
df_filtered = filter_data_per_member(df)
# Step 3: Slice the data into four separate DataFrames

del df

windows = [1, 5, 15, 25, 52]


def slice_data_dynamic(df_filtered, additional_shift,
                       windows=[1, 5, 15, 25, 52]):
    # Initialize dictionaries to hold the final DataFrames
    dfs = {weeks: [] for weeks in windows}

    for clab_id in df_filtered['CLAB_ID'].unique():
        member_df = df_filtered[df_filtered['CLAB_ID'] == clab_id]
        week_non_adhere = member_df['Week_non_adhere'].dropna().unique()[0]

        # additional_shift sets shift to determine which week we are predicting

        for weeks in windows:
            # Initialize the start and end week for
            # slicing with an additional shift
            week_start = 49 - weeks - additional_shift
            week_end = 49 - 1 - additional_shift

            # Slice the DataFrame for the current window with
            # an additional shift
            sliced_df = member_df[(member_df['Week'] >= week_start) &
                                  (member_df['Week'] <= week_end)]
            dfs[weeks].append(sliced_df)

    # Combine the sliced DataFrames for each window size
    for weeks in windows:
        dfs[weeks] = pd.concat(dfs[weeks])

    return [dfs[week] for week in windows]


def calculate_aggregate_stats(df, col_name):
    # Group the DataFrame by 'CLAB_ID' and calculate aggregates for
    # the specified column
    aggregates = df \
                    .groupby('CLAB_ID')[col_name] \
                    .agg(['min', 'max', 'mean', 'median', 'sum', 'std']) \
                    .reset_index()

    # Rename the columns to reflect the aggregate type for clarity
    aggregates.columns = ['CLAB_ID', f'{col_name}_min', f'{col_name}_max',
                          f'{col_name}_mean', f'{col_name}_median',
                          f'{col_name}_sum', f'{col_name}_std']

    # Fill NaN values in the 'std' column with 0
    aggregates[f'{col_name}_std'] = aggregates[f'{col_name}_std'].fillna(0)

    return aggregates


def calculate_aggregate_counts(df, col_name):
    # Perform one-hot encoding on the specified column
    one_hot_encoded = pd.get_dummies(df[[col_name, 'CLAB_ID']],
                                     columns=[col_name])

    # Aggregate the one-hot encoded DataFrame by
    # 'CLAB_ID', summing up the binary columns
    counts = one_hot_encoded.groupby('CLAB_ID').sum().reset_index()

    return counts


def calculate_static_values(df, col_name):
    # Generate one-hot encoded DataFrame
    one_hot_encoded = pd.get_dummies(df[[col_name, 'CLAB_ID']],
                                     columns=[col_name])

    # Group by 'CLAB_ID' and take the first occurrence
    static_values = one_hot_encoded.groupby('CLAB_ID').first().reset_index()

    # Convert boolean values to integers (True to 1, False to 0)
    for col in static_values.columns:
        if static_values[col].dtype == 'bool':
            static_values[col] = static_values[col].astype(int)

    return static_values


def calculate_aggregate_multicounts(df, col_name):
    # Explode the list into separate rows
    df_exploded = df.explode(col_name)

    # One-hot encode the exploded values
    one_hot_encoded = pd.get_dummies(df_exploded[[col_name, 'CLAB_ID']],
                                     columns=[col_name])

    # Aggregate the one-hot encoded DataFrame by 'CLAB_ID',
    # summing up the binary columns
    counts = one_hot_encoded.groupby('CLAB_ID').sum().reset_index()

    return counts


# For numerical columns, calculate rolling window features
numeric_columns = ['change_in_dc', 'AWPBASE_AMT', 'AWPCALC_AMT', 'DISPFEE_AMT',
                   'INGRCOST_AMT', 'INGRD_CHARGE_AMT', 'SALESTAX_AMT',
                   'RTL_PRC_AMT', 'DAYSSUPP_NBR', 'REFILL_CD']

static_columns = ['DRUGACCT_CD', 'CLMDRUGPLN_CD']

categorical_columns = [
    'DRUGGNRC_FL', 'ADMINRTE_CD', 'DOSEFORM_CD', 'MAILORD_IND', 'MAINT_CD',
    'MLTSGL_CD', 'DAW_CD', 'FORMULARY_IND', 'GNRCBRND_CD', 'OTC_CD',
    'COMPOUND_CD', 'PLANDRUGST_CD', 'DEA_CLASS_CD', 'PROC_CD',
    'UTILGRP_CD', 'NATPOS_DS', 'UTIL_DS', 'ER_IND'
]

multicat_columns = ['combined_diagnoses']

shifts = [0]

for shift in shifts:

    print(shift)

    # Apply the dynamic slicing
    df1, df4, df12, df21, df47 = slice_data_dynamic(df_filtered,
                                              additional_shift=shift)

    old_dfs = [df1, df4, df12, df21, df47]
    del df1, df4, df12, df21, df47
    new_dfs = [None, None, None, None, None]  # No need to declare df1a, df4a,
    # df12a, df48a separately

    # Use enumerate to get both the index and the old_df reference in the loop
    for i, old_df in enumerate(old_dfs):
        # Temporary variable to store the new DataFrame for
        # the current iteration
        temp_df = None

        # Numeric aggregations
        for col in numeric_columns:
            print(f'Generating {col} w{i}')
            agg_df = calculate_aggregate_stats(old_df,
                                               col)
            # Initialize temp_df with the first aggregation or
            # merge subsequent aggregations
            if temp_df is None:
                temp_df = agg_df
            else:
                temp_df = pd.merge(temp_df, agg_df,
                                   on='CLAB_ID',
                                   how='left')

        # Static values and Categorical counts, similar pattern as above
        for col in static_columns:
            print(f'Generating {col} w{i}')
            static_df = calculate_static_values(old_df, col)
            if temp_df is None:
                temp_df = static_df
            else:
                temp_df = pd.merge(temp_df, static_df,
                                   on='CLAB_ID',
                                   how='left')

        for col in categorical_columns:
            print(f'Generating {col} w{i}')
            counts_df = calculate_aggregate_counts(old_df, col)
            if temp_df is None:
                temp_df = counts_df
            else:
                temp_df = pd.merge(temp_df, counts_df,
                                   on='CLAB_ID',
                                   how='left')

        for col in multicat_columns:
            print(f'Generating {col} w{i}')
            multicounts_df = calculate_aggregate_multicounts(old_df, col)
            if temp_df is None:
                temp_df = multicounts_df
            else:
                temp_df = pd.merge(temp_df, multicounts_df,
                                   on='CLAB_ID',
                                   how='left')

        # Now, update the corresponding element in new_dfs directly
        new_dfs[i] = temp_df

    # At this point, new_dfs[0], new_dfs[1], new_dfs[2], and
    # new_dfs[3] contain the updated DataFrames
    df1a, df4a, df12a, df21a, df47a = new_dfs  # Now these variables
    # will hold the intended DataFrames

    # Merge dataframes horizontally
    # First, rename columns in df1, df4, df12,
    # df48 except for the common columns
    common_columns = ['CLAB_ID', 'days_from_index', 'Non_adhere',
                      'Week_non_adhere', 'Week']

    # First, define a function to rename columns with the specified suffix
    def rename_columns(df, suffix):
        # Rename all columns except for 'CLAB_ID' to include the suffix
        df = df.rename(columns={col: f"{col}_w{suffix}" if col != 'CLAB_ID'
        else col for col in df.columns})
        return df


    print('Rename')
    # Apply the renaming function to each DataFrame with the appropriate suffix
    df1r = rename_columns(df1a, windows[0])
    df4r = rename_columns(df4a, windows[1])
    df12r = rename_columns(df12a, windows[2])
    df21r = rename_columns(df21a, windows[3])
    df47r = rename_columns(df47a, windows[4])

    print('Merge')
    # Merge the renamed DataFrames, starting with df1
    # Note: We're excluding 'CLAB_ID' from the list for df4,
    # df12, and df48 to avoid duplicating the merge key
    dfs = [df1r, df4r, df12r, df21r, df47r]
    df_merged = reduce(lambda left, right: pd.merge(left,
                                                    right,
                                                    on='CLAB_ID',
                                                    how='outer'),
                       dfs)

    # Fill all missing values in the merged df to 0
    df_merged.fillna(0,inplace=True)

    df_merged['days_since_claim'] = temp

    print('Save')
    df_merged.to_csv(f'rxclaims_final.csv', index=False)
