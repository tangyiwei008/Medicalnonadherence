'''
Created on Feb 8 2024
By Otto Gaulke
Description: 
'''
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

print('Loading Procedure Data...')

df_proc = pd.read_csv('rxclaims_full.csv',
                      usecols=['PROC_CD'],
                      low_memory=False)

index_change = df_proc.copy()
index_change['PROC_CD'] = False
index_change = index_change['PROC_CD']

index = df_proc['PROC_CD'] == '@NVL'
index_change |= index

df_proc.loc[index, 'PROC_CD'] = np.nan

print('PROC 0%')

proc_dict1 = {'A': 'Administration of Vaccine',
              'F': 'Category II Codes',
              'M': 'Multianalyte Assay',
              'T': 'Category III Codes',
              'U': 'Proprietary Laboratory Analyses'}

index = df_proc['PROC_CD'].str[-1].isin(proc_dict1.keys())
index_change |= index

df_proc.loc[index, 'PROC_CD'] = df_proc \
                                    .loc[index, 'PROC_CD'] \
                                    .str[-1] \
                                    .map(proc_dict1)

print('PROC 16.5%')

proc_dict2 = {'M1141': 'Episode of Care',
              'M1142': 'Episode of Care',
              'M1143': 'Episode of Care'}

index = df_proc['PROC_CD'].str[0:5].isin(proc_dict2.keys())
index_change |= index

df_proc.loc[index, 'PROC_CD'] = df_proc \
                                    .loc[index, 'PROC_CD'] \
                                    .str[0:5] \
                                    .map(proc_dict2)

print('PROC 33%')

proc_dict3 = {'M000': 'MIPS Value Pathways',
              'M001': 'EOM Enhanced Services',
              'M110': 'Episode of Care',
              'M111': 'Episode of Care',
              'M112': 'Episode of Care',
              'M113': 'Episode of Care'}

index = df_proc['PROC_CD'].str[0:4].isin(proc_dict3.keys())
index_change |= index

df_proc.loc[index, 'PROC_CD'] = df_proc \
                                    .loc[index, 'PROC_CD'] \
                                    .str[0:4] \
                                    .map(proc_dict3)

print('PROC 50%')

proc_dict4 = {'A41': 'Skin Substitute Devices',
              'C10': 'Other Therapeutic Procedures',
              'C16': 'Surgical, Imaging Devices and Grafts',
              'M10': 'Screening Procedures',
              '992': 'Evaluation and Management Services',
              '993': 'Evaluation and Management Services',
              '994': 'Evaluation and Management Services'}

index = df_proc['PROC_CD'].str[0:3].isin(proc_dict4.keys())
index_change |= index

df_proc.loc[index, 'PROC_CD'] = df_proc \
                                    .loc[index, 'PROC_CD'] \
                                    .str[0:3] \
                                    .map(proc_dict4)

print('PROC 66.5')

proc_dict5 = {'A0': 'Ambulance and Other Transport Services and Supplies',
              'A2': 'Matrix for Wound Management',
              'A9': 'Administrative. Miscellaneous and Investigational',
              'J9': 'Chemotherapy Drugs',
              'K0': 'Durable medical equipment/Aministrative contractors',
              'K1': 'Components, Accessories and Supplies',
              'L0': 'Orthotic Procedures and services',
              'L1': 'Orthotic Procedures and services',
              'L2': 'Orthotic Procedures and services',
              'L3': 'Orthotic Procedures and services',
              'L4': 'Orthotic Procedures and services',
              'M0': 'Miscellaneous Medical Services',
              'M1': 'Other Services',
              'V2': 'Vision Services',
              'V5': 'Hearing Services'}

index = df_proc['PROC_CD'].str[0:2].isin(proc_dict5.keys())
index_change |= index

print('PROC 73%')

df_proc.loc[index, 'PROC_CD'] = df_proc \
                                    .loc[index, 'PROC_CD'] \
                                    .str[0:2] \
                                    .map(proc_dict5)

proc_dict6 = {'A': 'Medical And Surgical Supplies',
              'B': 'Enteral and Parenteral Therapy',
              'C': 'Outpatient PPS',
              'D': 'Dental Procedure and Services',
              'E': 'Durable Medical Equipment',
              'G': 'Procedures/Professional Services',
              'H': 'Alcohol and Drug Abuse Treatment',
              'J': 'Drugs Administered Other than Oral Method',
              'L': 'Prosthetic Procedures',
              'P': 'Pathology and Laboratory Services',
              'Q': 'Temporary Codes',
              'R': 'Diagnostic Radiology Services',
              'S': 'Temporary National Codes (Non-Medicare)',
              'T': 'National Codes Established for State Medicaid Agencies',
              'U': 'Coronavirus Diagnostic Panel',
              '0': 'Anesthesia',
              '1': 'Surgery',
              '2': 'Surgery',
              '3': 'Surgery',
              '4': 'Surgery',
              '5': 'Surgery',
              '6': 'Surgery',
              '7': 'Radiology Procedures',
              '8': 'Pathology and Laboratory Procedures',
              '9': 'Medicine Services and Procedures'}

index_change = ~index_change

print('PROC 100%')

df_proc.loc[index_change, 'PROC_CD'] = df_proc \
                                            .loc[index_change, 'PROC_CD'] \
                                            .str[0] \
                                            .map(proc_dict6)

print('Loading Additional Procedure Data...')

df_procsrg = pd.read_csv('rxclaims_full_new.csv',
                         usecols=['PROCSRG1_CD'],
                         low_memory=False)

procsrg_dict = {'0': 'Medical and Surgical',
                '1': 'Obstetrics',
                '2': 'Placement',
                '3': 'Administration',
                '4': 'Measurement and Monitoring',
                '5': 'Extracorporeal or Systemic Assistance and Performance',
                '6': 'Extracorporeal or Systemic Therapies',
                '7': 'Osteopathic',
                '8': 'Other Procedures',
                '9': 'Chiropractic',
                'B': 'Imaging',
                'C': 'Nuclear Imaging',
                'D': 'Radiation Therapy',
                'F': 'Physical Rehabilitation and Diagnostic Audiology',
                'G': 'Mental Health',
                'H': 'Substance Abuse',
                'X': 'New Technology'}

index = df_procsrg['PROCSRG1_CD'].str[0].isin(procsrg_dict.keys())

print('PROCSRG')

df_procsrg.loc[index, 'PROCSRG1_CD'] = df_procsrg \
                                            .loc[index, 'PROCSRG1_CD'] \
                                            .str[0] \
                                            .map(procsrg_dict)

index = ~pd.isna(df_procsrg['PROCSRG1_CD'])

print('Merging Procedures...')

df_proc.loc[index, 'PROC_CD'] = df_procsrg['PROCSRG1_CD']

df_diag = pd.DataFrame()

cols = ['ICD10_DIAGLN_CD', 'DIAG1_CD', 'DIAG2_CD',
        'DIAG3_CD', 'DIAG4_CD', 'DIAG5_CD']

for col in cols:
    print(f'Loading {col}...')

    df_diag_temp = pd.read_csv('rxclaims_full_new.csv',
                               usecols=[col],
                               low_memory=False)

    index_change = df_diag_temp.copy()
    index_change[col] = False
    index_change = index_change[col]

    diag_dict1 = {'R20': 'Symptoms/Signs of Skin/Subcutaneous Tissue',
                  'R21': 'Symptoms/Signs of Skin/Subcutaneous Tissue',
                  'R22': 'Symptoms/Signs of Skin/Subcutaneous Tissue',
                  'R23': 'Symptoms/Signs of Skin/Subcutaneous Tissue',
                  'R47': 'Symptoms/Signs of Speech/Voice',
                  'R48': 'Symptoms/Signs of Speech/Voice',
                  'R49': 'Symptoms/Signs of Speech/Voice',
                  'R80': 'Abnormal Findings of Urine, No Diagnosis',
                  'R81': 'Abnormal Findings of Urine, No Diagnosis',
                  'R82': 'Abnormal Findings of Urine, No Diagnosis',
                  'R97': 'Abnormal Tumor Markers',
                  'R99': 'Ill-defined/Unknown Cause of Mortality'}

    index = df_diag_temp[col].str[0:3].isin(diag_dict1.keys())
    index_change |= index

    df_diag_temp.loc[index, col] = df_diag_temp \
                                        .loc[index, col] \
                                        .str[0:3] \
                                        .map(diag_dict1)

    print(f'{col} 33%')

    diag_dict2 = {'D1': 'Neoplasms',
                  'D2': 'Neoplasms',
                  'D3': 'Neoplasms',
                  'D4': 'Neoplasms',
                  'H0': 'Diseases of the Eye and Adnexa',
                  'H1': 'Diseases of the Eye and Adnexa',
                  'H2': 'Diseases of the Eye and Adnexa',
                  'H3': 'Diseases of the Eye and Adnexa',
                  'H4': 'Diseases of the Eye and Adnexa',
                  'H5': 'Diseases of the Eye and Adnexa',
                  'R0': 'Symptoms/Signs of Circulatory/Respiratory Systems',
                  'R1': 'Symptoms/Signs of Digestive System/Abdomin',
                  'R2': 'Symptoms/Signs of Nervous/Musculoskeletal Systems',
                  'R3': 'Symptoms/Signs of Genitourinary System',
                  'R4': 'Symptom/Sign Cognition/Perception/Emotion Behavior',
                  'R5': 'General Signs and Symptoms',
                  'R6': 'General Signs and Symptoms',
                  'R7': 'Abnormal Findings of Blood, No Diagnosis',
                  'R8': 'Abnormal Findings of Other Fluid/Tissue No Diagnosis',
                  'R9': 'Abnormal Findings Imaging/Function, No Diagnosis',}

    index = df_diag_temp[col].str[0:2].isin(diag_dict2.keys())
    index_change |= index

    df_diag_temp.loc[index, col] = df_diag_temp \
                                        .loc[index, col] \
                                        .str[0:2] \
                                        .map(diag_dict2)

    print(f'{col} 66%')

    diag_dict3 = {'A': 'Certain Infectious and Parasitic Diseases',
                  'B': 'Certain Infectious and Parasitic Diseases',
                  'C': 'Neoplasms',
                  'D': 'Diseases of the Blood and Certain Immune Disorders',
                  'E': 'Endocrine, Nutritional and Metabolic Diseases',
                  'F': 'Mental, Behavioral, and Neurodevelopmental Disorders',
                  'G': 'Diseases Nervous System',
                  'H': 'Diseases Ear and Mastoid Process',
                  'I': 'Diseases Circulatory System',
                  'J': 'Diseases Respiratory System',
                  'K': 'Diseases Digestive System',
                  'L': 'Diseases Skin and Subcutaneous Tissue',
                  'M': 'Diseases Musculoskeletal System and Connective Tissue',
                  'N': 'Diseases Genitourinary System',
                  'O': 'Pregnancy, Childbirth and the Puerperium',
                  'P': 'Certain Conditions Originating in Perinatal Period',
                  'Q': 'Congenital Deformations/Chromosomal Abnormalities',
                  'S': 'Injury/Poisoning/Other Consequences External Causes',
                  'T': 'Injury/Poisoning/Other Consequences External Causes',
                  'U': 'Codes for Special Purposes',
                  'V': 'External Causes of Morbidity',
                  'X': 'External Causes of Morbidity',
                  'Y': 'External Causes of Morbidity',
                  'Z': 'Factors Influencing Health Status/Health Services'}

    index_change = ~index_change

    df_diag_temp.loc[index_change, col] = df_diag_temp \
                                                .loc[index_change, col] \
                                                .str[0] \
                                                .map(diag_dict3)

    print(f'{col} 100%')

    df_diag[col] = df_diag_temp[col]

chunks = pd.read_csv('rxclaims_full.csv',
                     chunksize=1000000,
                     low_memory=False)

print('Loading rxclaims Data...')

df = pd.DataFrame()

c = 25
for chunk in chunks:
    print(f'{c}%')
    df = pd.concat([df, chunk],
                   axis=0)
    c += 25

print('Dropping Columns...')

df.drop(columns=['ICD10_DIAGLN_DS', 'DIAG1_DS', 'PROC_DS',
                 'PROCSRG1_CD', 'NATPOS_CD', 'UTIL_CD',
                 'CCSR_CATEGORY_1', 'CCSR_CATEGORY_1_DESCRIPTION',
                 'CCSR_CATEGORY_2', 'CCSR_CATEGORY_2_DESCRIPTION',
                 'CCSR_CATEGORY_3', 'CCSR_CATEGORY_3_DESCRIPTION',
                 'CCSR_CATEGORY_4', 'CCSR_CATEGORY_4_DESCRIPTION'],
        inplace=True)

for col in cols:
    print(f'Replacing Recoded Column {col}...')
    df[col] = df_diag[col]

print('Replacing PROC_CD...')

df['PROC_CD'] = df_proc['PROC_CD']

print('Saving Data...')

df.to_csv('rxclaims_recoded.csv',
          index=False)
