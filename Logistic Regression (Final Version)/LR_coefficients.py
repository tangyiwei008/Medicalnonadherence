import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler


#training data
dfa = pd.read_csv('merged_final.csv', low_memory=False)
dfb = pd.read_csv('member_labels.csv', low_memory=False)
df1 = pd.merge(dfa,dfb,on = 'CLAB_ID',how = 'left')
#testing unseen data
df2 = pd.read_csv('merged_final_new_data.csv', low_memory=False)

#sensitive top 30 features
sensitive_features = pd.read_csv('feature_importance_full_final_sensitive.csv', low_memory=False)
sensitive_top_features = sensitive_features['Feature'].head(30).tolist()

scaler = MinMaxScaler()
# generate sensitive LR model and feature importance
X_train = df1[sensitive_top_features]

y_train = df1['Non_adhere_last4']

X_test = df2[sensitive_top_features]

scaled_model = scaler.fit(X_train)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

imputer_model = imputer.fit(X_train)

X_train = imputer_model.transform(X_train)

X_train = scaled_model.transform(X_train)

X_test = imputer_model.transform(X_test)

X_test = scaled_model.transform(X_test)

logreg_model_sensitive = LogisticRegression(
    class_weight={0: 1, 1: 4},
    C=10,
    penalty='elasticnet',
    l1_ratio=0.3,
    solver='saga'
)

logreg_model_sensitive.fit(X_train, y_train)


y_prob = logreg_model_sensitive.predict_proba(X_test)[:, 1]

# set threshold
new_threshold = 0.5
y_pred_sensitive = np.where(y_prob >= new_threshold, 1, 0)


coefficients1 = logreg_model_sensitive.coef_[0]
feature_names_sensitive = sensitive_top_features
fdf_s = pd.DataFrame({ 
    'Feature_sensitive': feature_names_sensitive,
    'Coefficients_sensitive': coefficients1
})
# Calculating the absolute values of the coefficients
fdf_s['Abs_Coefficients'] = fdf_s['Coefficients_sensitive'].abs()

# Sorting the DataFrame by the absolute values of the coefficients in descending order
fdf_s_sorted = fdf_s.sort_values(by='Abs_Coefficients', ascending=False)

# Saving the sorted DataFrame to a CSV file
fdf_s_sorted.to_csv('features_coefficients_sensitive.csv', index=False)
fdf_s_sorted[['Feature_sensitive', 'Coefficients_sensitive']].to_csv('coefficients_top30_features_sensitive.csv', index=False)



#full top 30 features
full_features = pd.read_csv('feature_importance_full_final.csv', low_memory=False)
full_top_features = full_features['Feature'].head(30).tolist()
# generate full LR model and feature importance
X_train = df1[full_top_features]

y_train = df1['Non_adhere_last4']

X_test = df2[full_top_features]

scaled_model = scaler.fit(X_train)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

imputer_model = imputer.fit(X_train)

X_train = imputer_model.transform(X_train)

X_train = scaled_model.transform(X_train)

X_test = imputer_model.transform(X_test)

X_test = scaled_model.transform(X_test)

logreg_model_full = LogisticRegression(
    class_weight={0: 1, 1: 4},
    C=10,
    penalty='elasticnet',
    l1_ratio=0.3,
    solver='saga'
)

logreg_model_full.fit(X_train, y_train)


y_prob = logreg_model_full.predict_proba(X_test)[:, 1]

# set threshold
new_threshold = 0.5
y_pred_full = np.where(y_prob >= new_threshold, 1, 0)


coefficients2 = logreg_model_full.coef_[0]
feature_names_full = full_top_features
fdf_f = pd.DataFrame({ 
    'Feature_full': feature_names_full,
    'Coefficients_full': coefficients2
})
# Calculating the absolute values of the coefficients
fdf_f['Abs_Coefficients'] = fdf_f['Coefficients_full'].abs()

# Sorting the DataFrame by the absolute values of the coefficients in descending order
fdf_f_sorted = fdf_f.sort_values(by='Abs_Coefficients', ascending=False)
fdf_f_sorted[['Feature_full', 'Coefficients_full']].to_csv('coefficients_top30_features.csv', index=False)
