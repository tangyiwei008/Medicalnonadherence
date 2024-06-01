import pandas as pd

# Read CLAB_MEMBER data
clab_member_df = pd.read_csv("/Users/emilyliu/Downloads/CAL/Rx Claim Data/CLAB_MEMBER.csv", encoding='utf-8')

# Read and concatenate rxclaims_1.csv to rxclaims_20.csv
rxclaims_dfs = []
for i in range(1, 21):
    filename = f"/Users/emilyliu/Downloads/CAL/Rx Claim Data/rxclaims_{i}.csv"
    try:
        rxclaims_dfs.append(pd.read_csv(filename, encoding='utf-8'))
    except UnicodeDecodeError:
        # If 'utf-8' encoding fails, try other common encodings
        try:
            rxclaims_dfs.append(pd.read_csv(filename, encoding='latin-1'))
        except UnicodeDecodeError:
            rxclaims_dfs.append(pd.read_csv(filename, encoding='iso-8859-1'))

rxclaims_concatenated_df = pd.concat(rxclaims_dfs)

# Merge CLAB_MEMBER with concatenated rxclaims data based on CLAB_ID
merged_df = pd.merge(clab_member_df, rxclaims_concatenated_df, on='CLAB_ID', how='inner')

#0 <=week nonadherence *7 - days from index <= 180
# Impute null values in "Week_non_adhere" with integer 18
merged_df['Week_non_adhere'].fillna(18, inplace=True)

# Convert "Week_non_adhere" to integer type
merged_df['Week_non_adhere'] = merged_df['Week_non_adhere'].astype(int)

# Filter rows based on the condition
filtered_df = merged_df[(48 * 7 - merged_df['days_from_index'] >= 0) &
                        (48 * 7 - merged_df['days_from_index'] <= 180)]
filtered_df['weeks_from_index'] = filtered_df['days_from_index'] // 7

# Now merged_df contains the combined data
#Flag 1
filtered_df['MAILORD_PERCENTAGE'] = filtered_df['MAILORD_IND'].map({'Y': 1, 'N': 0})
MAILORD_Series = filtered_df.groupby('CLAB_ID')['MAILORD_PERCENTAGE'].mean().round(2)
MAILORD = MAILORD_Series.reset_index()

# Applying conditions to set values in column 'K' 
import numpy as np

# Define conditions and corresponding values
conditions = [
    (MAILORD['MAILORD_PERCENTAGE'] == 0),
    (MAILORD['MAILORD_PERCENTAGE'] == 1),
    (MAILORD['MAILORD_PERCENTAGE'] > 0) & (MAILORD['MAILORD_PERCENTAGE'] < 0.75),
    (MAILORD['MAILORD_PERCENTAGE'] >= 0.75) & (MAILORD['MAILORD_PERCENTAGE'] < 1)
]
values = ['None', 'All', 'Some', 'Most']

# Apply conditions using numpy.select()
MAILORD['MAILORD'] = np.select(conditions, values, default='Some')

#Flag2 
# Filter the DataFrame where UTIL_CD is equal to RX0001
filtered_df_maintaince_drug = filtered_df[filtered_df['MAINT_DS'] == 'Maintenance Drug']

def custom_agg(group): 
    drug_count = group['DRUG_NM'].count()
    total_visit_count = group['weeks_from_index'].nunique()
    return drug_count / total_visit_count if total_visit_count else 0 # Performing the groupby and custom aggregation 

avg_drug_count_per_visit = filtered_df_maintaince_drug.groupby('CLAB_ID').apply(custom_agg).reset_index(name='AVERAGE_DRUG_COUNT_PER_PHARMACY_VISIT').round(2)

#Flag3
# Count unique values in "DRUG_NM" column for each unique value in "CLAB_ID"
drug_count_per_member= filtered_df_maintaince_drug.groupby('CLAB_ID')['DRUG_NM'].nunique()
drug_count_per_member = drug_count_per_member.reset_index(name='DRUG_COUNT_PER_MEMBER')


#Flag4
# Calculate cost_per_day
filtered_df['AVERAGE_DRUG_COST_PER_DAY'] = filtered_df['RTL_PRC_AMT'] / filtered_df['DAYSSUPP_NBR']
# Group by CLAB_ID and calculate the total cost_per_day
top_20_cost = filtered_df.groupby('CLAB_ID')['AVERAGE_DRUG_COST_PER_DAY'].sum().round(2)
top_20_cost = top_20_cost.reset_index()
# Calculate threshold for top 20%
threshold = top_20_cost['AVERAGE_DRUG_COST_PER_DAY'].quantile(0.8)
# Label CLAB_ID based on top 20% or not
top_20_cost['IN_TOP_20%_OF_DRUG_COST?'] = top_20_cost['AVERAGE_DRUG_COST_PER_DAY'].apply(lambda x: 'Yes' if x >= threshold else 'No')


# Merge dataframes based on the 'CLAB_ID' column
add_predictions = pd.read_csv('/Users/emilyliu/Downloads/CAL/Rx Claim Data/predictions.csv')
df = add_predictions.merge(MAILORD, on='CLAB_ID', how='inner')
# Rename the merged column to 'PREDICTION'
df.rename(columns={'Max_Prediction': 'PREDICTION'}, inplace=True)
# Convert the values in the 'PREDICTION' column to percentages
df['PREDICTION'] = (df['PREDICTION'] * 100).apply(lambda x: "{:.0f}%".format(x))
df = df.merge(avg_drug_count_per_visit, on='CLAB_ID', how='left')
df = df.merge(drug_count_per_member, on='CLAB_ID', how='left')
df = df.merge(top_20_cost, on='CLAB_ID', how='left')
add_sdoh_models_df = pd.read_csv('/Users/emilyliu/Downloads/CAL/Rx Claim Data/ADD_SDOH_MODELS.csv')
# Merge add_sdoh_models_df with df
df = df.merge(add_sdoh_models_df, on='CLAB_ID', how='left')
# Write merged dataframe to CSV
df['AVERAGE_DRUG_COST_PER_DAY'] = df['AVERAGE_DRUG_COST_PER_DAY'].apply(lambda x: "${:,.2f}".format(x))
df['MAILORD_PERCENTAGE'] = (df['MAILORD_PERCENTAGE'] * 100).apply(lambda x: "{:.0f}%".format(x))
df.rename(columns={'IPRO_RISK_LEVEL_FINANCIAL': 'RISK_LEVEL_FINANCIAL'}, inplace=True)
df.rename(columns={'IPRO_RISK_LEVEL_FOOD': 'RISK_LEVEL_FOOD'}, inplace=True)
df.rename(columns={'IPRO_RISK_LEVEL_HOUSING': 'RISK_LEVEL_HOUSING'}, inplace=True)
df.rename(columns={'IPRO_RISK_LEVEL_SOCIAL': 'RISK_LEVEL_SOCIAL'}, inplace=True)
df.rename(columns={'IPRO_RISK_LEVEL_TRANSPORTATION': 'RISK_LEVEL_TRANSPORTATION'}, inplace=True)
df.to_csv('final_output.csv', index=False)
