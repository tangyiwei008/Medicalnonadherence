import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option('display.min_rows', 20)


df = pd.read_csv('D:/School 2.0 Data/CAL EL Project BCBS/merged/merged_model_unbiased_4week.csv', low_memory=False)

df = df[~df['Non_adhere'].isna()]

# Drop Weeks 1-5 to ensure an adequate period of adherence (30+ days) for all members in the model
df = df[df['Week_non_adhere'] > 5]

df.drop(columns=['Week_non_adhere', 'CLAB_ID'], inplace=True)
y = df['Non_adhere']
X = df.loc[:, df.columns != 'Non_adhere']

top_200 = pd.read_csv('top_200_features_4w.csv')
cols_to_keep = list(top_200['Feature'].values)

# bugfix (xgboost is throwing errors due to these columns)
bugged_cols = ['CLMDRUGPLN_CD_D093900R_w1', 'CLMDRUGPLN_CD_D093600R_w1', 'DRUGACCT_CD_MNH5959014_w1',
               'CLMDRUGPLN_CD_D093900R_w5', 'DRUGACCT_CD_MNH5959009_w5']

for b in bugged_cols:
    if b in cols_to_keep:
        cols_to_keep.remove(b)

X = X[cols_to_keep]

# Drop columns that are directly related to outcome variable (adherence).
off_limits = ['cas_nums_172', 'cas_nums_10119', 'cas_nums_39', 'cas_nums_49', 'cas_nums_17', 'cas_nums_23',
              'cas_nums_24', 'cas_nums_25', 'cas_nums_26', 'cas_nums_29', 'cas_nums_50', 'cas_nums_51', 'cas_nums_40',
              'cas_nums_41', 'cas_nums_48', 'cas_nums_120']

# Filter cols_to_keep to only include columns not containing any of the bad roots
cols_to_drop2 = [col for col in cols_to_keep if any(bad_root in col for bad_root in off_limits)]
X.drop(columns=cols_to_drop2, inplace=True)

# Also drop race-sensitive features
race_sensitive = ['RACE', 'ETHNICITY', 'COUNTRY']
cols_to_drop3 = [col for col in cols_to_keep if any(bad_root in col for bad_root in race_sensitive)]
X.drop(columns=cols_to_drop3, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SimpleImputer to impute missing values using the mean
fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the training data
fill_NaN.fit(X_train)

# Transform both training and testing data using the fitted imputer
X_imp_train = pd.DataFrame(fill_NaN.transform(X_train))
X_imp_test = pd.DataFrame(fill_NaN.transform(X_test))

# Restore the original column names and indices to the imputed dataframes
X_imp_train.columns = X_train.columns
X_imp_test.columns = X_test.columns
X_imp_train.index = X_train.index
X_imp_test.index = X_test.index

# Update the original variables
X_train = X_imp_train
X_test = X_imp_test

# Standardization
non_binary_columns = [col for col in X_train.columns if not all(X_train[col].isin([0.0, 1.0, 0, 1]))]

# Initialize the StandardScaler
# scaler = StandardScaler()
scaler = MinMaxScaler()

# Fit the scaler on non-binary columns of the training data
scaler.fit(X_train[non_binary_columns])

# Transform both training and testing non-binary columns using the fitted scaler
X_train[non_binary_columns] = scaler.transform(X_train[non_binary_columns])
X_test[non_binary_columns] = scaler.transform(X_test[non_binary_columns])

# Initialize Logistic Regression model
logistic_model = LogisticRegression(penalty='l1', solver='saga', max_iter=999999, random_state=42)

# Fit the model on the training data
logistic_model.fit(X_train, y_train)

# Predictions
y_pred = logistic_model.predict(X_test)
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

# Calculating accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculating ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc}")

# Generating a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Generating a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Extracting coefficients
coefficients = logistic_model.coef_[0]  # For binary classification, .coef_ returns a 2D array

# Mapping coefficients to feature names
feature_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

# Sorting the features by the absolute value of their coefficient in descending order
top_coefficients = feature_coefficients.reindex(feature_coefficients.Coefficient.abs().sort_values(ascending=False).index)

print("Coefficients:")
print(top_coefficients)

top_coefficients.to_csv('top_coefficients_lr.csv', index=False)

# Plot Learning Curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Plotting the learning curve
title = "Learning Curve"
# Cross validation with 5 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
plot_learning_curve(logistic_model, title, X, y, ylim=(0.7, 1.01), cv=5, n_jobs=4)
plt.show()


# Plot learning curve with log loss
def plot_learning_curve2(estimator, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                         scoring='neg_log_loss'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)

    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Log Loss")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


# Now you can call the function with your estimator and data
plot_learning_curve2(logistic_model, X_train, y_train, cv=5, n_jobs=-1, scoring='neg_log_loss')
plt.show()

# Plot ROC curve
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
