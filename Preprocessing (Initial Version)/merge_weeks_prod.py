"""
Created on Feb 11 2024
By Spencer Stromback
Description: Aggregates weekly data into a single row per member.
This is to be combined with claims data (after similar compression)
This program prepares the data for the 4-week prediction model.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from functools import reduce

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Setting random seed for reproducibility
np.random.seed(42)

dfs = [pd.read_csv(f'weeks_{n}..csv', low_memory=False) for n in range(1, 11)]

df = pd.concat(dfs, ignore_index=True)

df.rename(columns={'Weekcount':'Week'}, inplace=True)


def unpack_cols(row):
    row = row[1:-1]
    row = row.split(',')
    return row


# Reformat list columns into actual lists
df.risk_nums = df.risk_nums.apply(unpack_cols)
df.cas_nums = df.cas_nums.apply(unpack_cols)
df.cln_nums = df.cln_nums.apply(unpack_cols)


# *DO WE WANT TO DROP AGE HERE, OR FROM THE OTHER SHEET?*
columns_to_drop = [
    'opp_nums', 'line_id', 'mbr_age_nbr'
]

# Drop columns
df.drop(columns=columns_to_drop, inplace=True)

df.sort_values(by=['CLAB_ID', 'Week'],
               ascending=[True, False],
               inplace=True)

# Create Week_non_adhere column, then drop later
df['Week_non_adhere'] = 53

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
            week_start = week_non_adhere - 52
            week_end = week_non_adhere - 1
            # Filter the member's data for the 52 weeks prior to
            # Week_non_adhere
            member_filtered = member_df[(member_df['Week'] >= week_start) &
                                        (member_df['Week'] <= week_end)]
            filtered_dfs.append(member_filtered)

    return pd.concat(filtered_dfs)  # Combine all filtered DataFrames into one


# Apply the filtering
df_filtered = filter_data_per_member(df)
# Step 3: Slice the data into four separate DataFrames

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
            # Initialize the start and end week for slicing with
            # an additional shift
            week_start = week_non_adhere - weeks - additional_shift
            week_end = week_non_adhere - 1 - additional_shift

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
    aggregates = df.groupby('CLAB_ID')[col_name] \
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

    # Aggregate the one-hot encoded DataFrame by 'CLAB_ID',
    # summing up the binary columns
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


def calculate_list_counts(df, col_name):
    # Filter relevant columns and drop rows where col_name is NaN
    relevant_df = df[['CLAB_ID', col_name]].dropna(subset=[col_name])

    # Initialize MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer(sparse_output=True)

    # Perform one-hot encoding on col_name
    encoded_data = mlb.fit_transform(relevant_df[col_name])

    # Create a DataFrame from the encoded sparse matrix
    encoded_df = pd.DataFrame.sparse.from_spmatrix(
        encoded_data,
        index=relevant_df.index,
        columns=[f'{col_name}_{cls}' for cls in mlb.classes_]
    )

    # Concatenate CLAB_ID back with the encoded DataFrame
    encoded_df['CLAB_ID'] = relevant_df['CLAB_ID'].values

    # Group by 'CLAB_ID' and aggregate with 'sum' to count occurrences
    sum_df = encoded_df.groupby('CLAB_ID').sum()

    # Determine presence (acting as 'max') by checking if sum is greater than 0
    presence_df = (sum_df > 0).astype(int)

    # Prepare final DataFrame with both 'sum' and 'presence' information
    # Since 'sum_df' already contains the sum, we only need to
    # rename columns for 'presence_df'
    presence_df.columns = [f'{col}_presence' for col in presence_df.columns]

    # Combine 'sum' and 'presence' DataFrames
    final_df = sum_df.join(presence_df, rsuffix='_max')

    # Reset index to make 'CLAB_ID' a column again in the final DataFrame
    final_df.reset_index(inplace=True)

    # Flatten the DataFrame columns if necessary and prepare for
    # merging or further processing
    final_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else
                        col for col in final_df.columns.values]

    return final_df


# For numerical columns, calculate rolling window features
numeric_columns = [
    'clinical_risk_score', 'er_visit_ci_count', 'inpatient_ci_count'
]

static_columns = [

]

categorical_columns = [
    'cust_care_hlth_sys', 'health_cat', 'No_attrib'
]

list_columns = [
    'risk_nums', 'cas_nums', 'cln_nums'
]

shifts = [0]

for shift in shifts:

    # Apply the dynamic slicing
    df1, df4, df12, df21, df47 = slice_data_dynamic(df_filtered,
                                              additional_shift=shift)

    old_dfs = [df1, df4, df12, df21, df47]
    new_dfs = [None, None, None, None, None]
    # No need to declare df1a, df4a, df12a, df48a separately

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
                temp_df = pd.merge(temp_df,
                                   agg_df,
                                   on='CLAB_ID',
                                   how='left')

        # Static values and Categorical counts, similar pattern as above
        for col in static_columns:
            print(f'Generating {col} w{i}')
            static_df = calculate_static_values(old_df,
                                                col)
            if temp_df is None:
                temp_df = static_df
            else:
                temp_df = pd.merge(temp_df,
                                   static_df,
                                   on='CLAB_ID',
                                   how='left')

        for col in categorical_columns:
            print(f'Generating {col} w{i}')
            counts_df = calculate_aggregate_counts(old_df, col)
            if temp_df is None:
                temp_df = counts_df
            else:
                temp_df = pd.merge(temp_df,
                                   counts_df,
                                   on='CLAB_ID',
                                   how='left')

        for col in list_columns:
            print(f'Generating {col} w{i}')
            lists_df = calculate_list_counts(old_df,
                                             col)
            if temp_df is None:
                temp_df = lists_df
            else:
                temp_df = pd.merge(temp_df,
                                   lists_df,
                                   on='CLAB_ID',
                                   how='left')

        # Now, update the corresponding element in new_dfs directly
        new_dfs[i] = temp_df

    # At this point, new_dfs[0], new_dfs[1], new_dfs[2],
    # and new_dfs[3] contain the updated DataFrames
    df1a, df4a, df12a, df21a, df47a = new_dfs  # Now these variables will hold
    # the intended DataFrames

    # Merge dataframes horizontally

    # First, define a function to rename columns with the specified suffix
    def rename_columns(df, suffix):
        # Rename all columns except for 'CLAB_ID' to include the suffix
        df = df.rename(columns={col: f"{col}_w{suffix}" if col != 'CLAB_ID' else col for col in df.columns})
        return df


    # Apply the renaming function to each DataFrame with the appropriate suffix
    df1r = rename_columns(df1a, windows[0])
    df4r = rename_columns(df4a, windows[1])
    df12r = rename_columns(df12a, windows[2])
    df21r = rename_columns(df21a, windows[3])
    df47r = rename_columns(df47a, windows[4])

    # Merge the renamed DataFrames, starting with df1
    # Note: We're excluding 'CLAB_ID' from the list for df4,
    # df12, and df48 to avoid duplicating the merge key
    dfs = [df1r, df4r, df12r, df21r, df47r]
    df_merged = reduce(lambda left, right: pd.merge(left,
                                                    right,
                                                    on='CLAB_ID',
                                                    how='outer'), dfs)

    # Fill all missing values in the merged df to 0
    df_merged.fillna(0,
                     inplace=True)

    df_merged.to_csv(f'ipro_prod.csv',
                     index=False)
