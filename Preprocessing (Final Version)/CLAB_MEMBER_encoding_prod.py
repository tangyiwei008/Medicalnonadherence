"""
Created on Feb 12 2024
By Spencer Stromback
Description: Encodes CLAB_MEMBER file into a usable format.
Output is usable for both 1 and 4 week prediction models.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Setting random seed for reproducibility
np.random.seed(42)

# Load the data
df = pd.read_csv('CLAB_MEMBER.csv',
                 low_memory=False)

df_county = pd.read_csv('Member_CNTY.csv',
                        low_memory=False)

df_sdohflags = pd.read_csv('ADD_SDOH_MODELS.csv',
                           low_memory=False)

# Add member county data
df = pd.merge(df,
              df_county,
              how='left',
              on='CLAB_ID')

# Add SDOH model flags
df = pd.merge(df,
              df_sdohflags,
              how='left',
              on='CLAB_ID')

# Drop FIPS codes
df.drop(columns=['FIPS_CNTY_CD'],
        inplace=True)

# Sort the DataFrame by 'CLAB_ID' and 'Non_Adhere' descending,
# so Non_Adhere=1 comes first
df = df.sort_values(by=['CLAB_ID', 'Non_Adhere'],
                    ascending=[True, False])

# Drop duplicates, keeping the first occurrence where
# Non_Adhere is likely 1 if it exists
df = df.drop_duplicates(subset='CLAB_ID', keep='first')

# Convert the 'Week_non_adhere' column to integers
df['Week_non_adhere'] = df['Week_non_adhere'].astype(int)

# Recode Race column (new name 'sdoh_race_recode')
# Code from Sam at BCBS
df['sdoh_race_recode'] = df['FINAL_RACE']

index_sor = df['FINAL_RACE_MULTIPLE'] \
                .str \
                .contains('SOME OTHER RACE', na=False)

index_asian = df['FINAL_RACE_MULTIPLE'] \
                    .str \
                    .contains('ASIAN', na=False)

index_boaa = df['FINAL_RACE_MULTIPLE'] \
                    .str \
                    .contains('BLACK OR AFRICAN AMERICAN', na=False)

index_aian = df['FINAL_RACE_MULTIPLE'] \
                    .str \
                    .contains('AMERICAN INDIAN OR ALASKAN NATIVE', na=False)

index_nopi = df['FINAL_RACE_MULTIPLE'] \
                    .str \
                    .contains('NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
                              na=False)

df.loc[index_sor, 'sdoh_race_recode'] = 'SOME OTHER RACE'
df.loc[index_asian, 'sdoh_race_recode'] = 'ASIAN'
df.loc[index_boaa, 'sdoh_race_recode'] = 'BLACK OR AFRICAN AMERICAN'
df.loc[index_aian, 'sdoh_race_recode'] = 'AMERICAN INDIAN OR ALASKAN NATIVE'
df.loc[index_nopi,
       'sdoh_race_recode'] = 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'


# Recode language column (new name 'english_speaker')
def recode_lan(row):
    if (pd.isna(row['FINAL_LANGUAGE_PRIMARY']) and
            pd.isna(row['FINAL_LANGUAGE_MULTIPLE'])):

        return row['FINAL_LANGUAGE_PRIMARY']

    else:

        if 'ENGLISH' in row['FINAL_LANGUAGE_PRIMARY']:

            return 'ENGLISH'

        elif pd.isna(row['FINAL_LANGUAGE_MULTIPLE']):

            return row['FINAL_LANGUAGE_PRIMARY']

        else:

            if 'ENGLISH' in row['FINAL_LANGUAGE_MULTIPLE']:

                return 'ENGLISH'

            else:

                return 'NON_ENGLISH'


df['english_speaker'] = df.apply(recode_lan,
                                 axis=1)


# Recode country of origin (new name 'sdoh_country_recode')
def recode_country(row):
    if (pd.isna(row['FINAL_COUNTRY_OF_ORIGIN']) and
            pd.isna(row['FINAL_COUNTRY_OF_ORIGIN_MULTIPLE'])):

        return row['FINAL_COUNTRY_OF_ORIGIN']

    else:

        if not pd.isna(row['FINAL_COUNTRY_OF_ORIGIN']):

            if row['FINAL_COUNTRY_OF_ORIGIN'] == 'MULTI':

                s = row['FINAL_COUNTRY_OF_ORIGIN_MULTIPLE'].split(' | ')

                if 'UNITED STATES' in s[0] or 'OTHER' in s[0]:

                    return s[1]

                elif 'UNITED STATES' in s[1] or 'OTHER' in s[1]:

                    return s[0]

                else:

                    return s[0]


df['sdoh_country_recode'] = df.apply(recode_country,
                                     axis=1)


# Cut off Age
def recode_age(row):
    if row < 60:

        return 59

    elif row >= 60 and row < 65:

        return 64

    elif row >= 65 and row < 70:

        return 69

    elif row >= 70 and row < 75:

        return 74

    elif row >= 75 and row < 80:

        return 79

    elif row >= 80 and row < 85:

        return 84

    elif row >= 85:

        return 85


df['Age'] = df['Age'].apply(recode_age)

df = df.drop(columns=['FINAL_RACE', 'FINAL_RACE_MULTIPLE',
                      'FINAL_LANGUAGE_PRIMARY', 'FINAL_LANGUAGE_MULTIPLE',
                      'FINAL_COUNTRY_OF_ORIGIN',
                      'FINAL_COUNTRY_OF_ORIGIN_MULTIPLE'])

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore', drop='first')

categorical_cols = ['CNTY_NM', 'IPRO_RISK_LEVEL_FINANCIAL',
                    'IPRO_RISK_LEVEL_FOOD', 'IPRO_RISK_LEVEL_HOUSING',
                    'IPRO_RISK_LEVEL_SOCIAL',
                    'IPRO_RISK_LEVEL_TRANSPORTATION',
                    'GENDER_AS_SUBMITTED', 'MEMBER_MARITAL_STATUS', 'FINAL_ETHNICITY_HISPANIC',
                    'RURAL_URBAN_CODE', 'sdoh_race_recode', 'english_speaker',
                    'sdoh_country_recode', 'USDA_FD_TRACT_LILA_1AND10_F',
                    'USDA_FD_TRACT_LILA_HALFAND10_F', 'USDA_FD_TRACT_LILA_1AND20_F',
                    'USDA_FD_TRACT_LILA_VEHICLE_F', 'USDA_FD_TRACT_HUNV_F',
                    'USDA_FD_TRACT_LOWINCOME_F', 'USDA_FD_TRACT_LA_1AND10_F',
                    'USDA_FD_TRACT_LA_HALFAND10_F', 'USDA_FD_TRACT_LA_1AND20_F',
                    'USDA_FD_TRACT_LA_HALF_F', 'USDA_FD_TRACT_LA_1_F', 'USDA_FD_TRACT_LA_10_F',
                    'USDA_FD_TRACT_LA_20_F', 'USDA_FD_TRACT_LA_VEHICLE_20_F'
    ]

df_encoded = pd.DataFrame(
    encoder.fit_transform(df[categorical_cols]).toarray(),
    columns=encoder.get_feature_names_out(categorical_cols))

df = df.drop(categorical_cols,
             axis=1)

df = pd.concat([df, df_encoded],
               axis=1)

# Imputation
fill_NaN = SimpleImputer(missing_values=np.nan,
                         strategy='mean')

df_imp = pd.DataFrame(fill_NaN.fit_transform(df))

df_imp.columns = df.columns

df_imp.index = df.index

print('Saving File...')

df_imp.to_csv('MEMBER_enc.csv',
              index=False)
