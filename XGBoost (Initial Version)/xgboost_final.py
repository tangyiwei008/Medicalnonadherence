import numpy as np
import pandas as pd
import xgboost
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import shap
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


# Function to tune hyperparameters of XGBoost using Bayesian Optimization, adapted for binary classification
def bo_tune_xgb(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree, learning_rate, alpha,
                reg_lambda, X, y):
    # Setting the parameters for XGBoost
    params = {'n_estimators': int(round(n_estimators)),
              'max_depth': int(round(max_depth)),
              'gamma': gamma,
              'min_child_weight': min_child_weight,
              'subsample': subsample,
              'colsample_bytree': colsample_bytree,
              'learning_rate': learning_rate,
              'alpha': alpha,
              'lambda': reg_lambda,
              'use_label_encoder': False,
              'eval_metric': 'logloss'}

    # Initializing the XGBoost classifier with given parameters
    xg_clf = xgb.XGBClassifier(**params, random_state=42)

    # Cross-validation to evaluate the model
    cv_scores = cross_val_score(xg_clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)

    # Calculating the mean accuracy
    mean_accuracy = cv_scores.mean()
    return mean_accuracy


# Setting up Bayesian Optimization with the specified parameter bounds, adapted for binary classification
xgb_bo = BayesianOptimization(
    f=lambda n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree, learning_rate, alpha,
             reg_lambda:
    bo_tune_xgb(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree, learning_rate, alpha,
                reg_lambda, X, y),
    # pbounds={'n_estimators': (50, 300),
    #          'max_depth': (3, 10),
    #          'gamma': (0, 5),
    #          'min_child_weight': (1, 6),
    #          'subsample': (0.5, 1.0),
    #          'colsample_bytree': (0.5, 1.0),
    #          'learning_rate': (0.01, 0.3),
    #          'alpha': (0, 1),
    #          'reg_lambda': (1, 4)},
    pbounds={'n_estimators': (50, 300),
             'max_depth': (3, 7),  # Smaller max_depth to reduce complexity
             'gamma': (1, 5),  # Higher gamma for more aggressive pruning
             'min_child_weight': (3, 10),  # Higher values to control over-fitting
             'subsample': (0.5, 0.8),  # Less than 1.0 to use less data
             'colsample_bytree': (0.3, 0.7),  # Less than 1.0 to use less features
             'learning_rate': (0.01, 0.2),  # Smaller learning rate for more conservative updates
             'alpha': (1, 10),  # Higher alpha for L1 regularization
             'reg_lambda': (2, 10)},  # Higher lambda for L2 regularization
    random_state=1, )

# Running the optimization process
xgb_bo.maximize(n_iter=5, init_points=8)

# Extracting the best parameters from the optimization, adapted for binary classification
best_params = xgb_bo.max['params']
best_params['n_estimators'] = int(round(best_params['n_estimators']))
best_params['max_depth'] = int(round(best_params['max_depth']))

# Training the XGBoost model with optimized parameters, using XGBClassifier for binary outcome
optimized_xgb = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
optimized_xgb.fit(X, y)

# Making predictions on the test dataset
y_pred = optimized_xgb.predict(X_test)
y_pred_proba = optimized_xgb.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

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

# Extracting the best parameters from the optimization, adapted for binary classification
best_params = xgb_bo.max['params']

# Converting any float hyperparameters to integers where necessary (e.g., 'n_estimators', 'max_depth')
best_params['n_estimators'] = int(round(best_params['n_estimators']))
best_params['max_depth'] = int(round(best_params['max_depth']))

# Print the optimal hyperparameters
print("Optimal hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

feature_importance = optimized_xgb.feature_importances_

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display feature importance
print(importance_df)
importance_df.to_csv('feature_importance.csv', index=False)


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


# Training the XGBoost model with optimized parameters, using XGBClassifier for binary outcome
optimized_xgb = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)

# Plotting the learning curve
title = "Learning Curve"
# Cross validation with 5 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
plot_learning_curve(optimized_xgb, title, X, y, ylim=(0.7, 1.01), cv=5, n_jobs=4)
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


def plot_shap_waterfall(model, X, instance_index=0, max_display=15):
    # Initialize SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Create a new figure for plotting
    plt.figure()

    # Generate waterfall plot for the specified instance
    # Set 'show=False' to prevent immediate display
    shap.plots.waterfall(shap_values[instance_index], max_display=max_display, show=False)

    # Save the figure
    plt.savefig('waterfall.png', bbox_inches='tight')

    # Now show the plot
    plt.show()


# Now you can call the function with your estimator and data
plot_learning_curve2(optimized_xgb, X_train, y_train, cv=5, n_jobs=-1, scoring='neg_log_loss')
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

# Assuming this is where you train your model
optimized_xgb.fit(X_train, y_train)

# Immediately confirm by checking if the model has been fitted
try:
    print("Model training completed. Number of boosting rounds:", len(optimized_xgb.get_booster().get_dump()))
except xgboost.core.XGBoostError as e:
    print("Error confirming model training:", e)

# Right after confirming the model is trained, attempt a prediction
try:
    # Attempting a prediction on a small subset to confirm the model is ready
    pred = optimized_xgb.predict(X_test.iloc[:1, :])
    print("Prediction on a small subset:", pred)
except Exception as e:
    print("Error making a prediction:", e)

# Example check: Attempt a prediction to confirm the model is fitted
try:
    print(optimized_xgb.predict(X_test.iloc[:1, :]))
except xgboost.core.XGBoostError as e:
    print("Model is not fitted. Error:", e)
    # Ensure the correct fitted model is being used

if len(X_test) > 0:  # Check if X_test is not empty
    plot_shap_waterfall(optimized_xgb, X_test, instance_index=0, max_display=15)

y_pred_proba = optimized_xgb.predict_proba(X_test)[:, 1]

# Test print
y_pred_proba_series = pd.Series(y_pred_proba).round(1)
print(y_pred_proba_series.value_counts())

final_predictions = y_pred_proba
final_binary_predictions = (y_pred_proba >= 0.5).astype(int)

# Generate Predictive Results for top 10% of Predictions
# Assuming y_pred_proba contains the predicted probabilities for the positive class
# Convert predicted probabilities to binary predictions based on a threshold (e.g., 0.5)
binary_predictions = (y_pred_proba >= 0.5).astype(int)

# Reset the index of y_test for alignment
y_test_reset = y_test.reset_index(drop=True)

# Identify indices of positive and negative predictions
positive_prediction_indices = np.where(binary_predictions == 1)[0]
negative_prediction_indices = np.where(binary_predictions == 0)[0]

# For positive predictions, find the ones with the highest probability (most confident)
top_10_positive_confidence_indices = positive_prediction_indices[np.argsort(y_pred_proba[positive_prediction_indices])[-len(positive_prediction_indices) // 10:]]

# For negative predictions, find the ones with the lowest probability (most confidently negative)
top_10_negative_confidence_indices = negative_prediction_indices[np.argsort(y_pred_proba[negative_prediction_indices])[:len(negative_prediction_indices) // 10]]

# Extract the top 10% predictions and true labels for each group
y_true_top_10_positive = y_test_reset.iloc[top_10_positive_confidence_indices]
predictions_top_10_positive = y_pred_proba[top_10_positive_confidence_indices]

y_true_top_10_negative = y_test_reset.iloc[top_10_negative_confidence_indices]
predictions_top_10_negative = y_pred_proba[top_10_negative_confidence_indices]

# Evaluate and print metrics for each group
print("Top 10% Most Confident Positive Predictions:")
print("Accuracy:", accuracy_score(y_true_top_10_positive, binary_predictions[top_10_positive_confidence_indices]))
# Assuming binary classification, if needed, calculate and print additional metrics

print("\nTop 10% Most Confident Negative Predictions:")
print("Accuracy:", accuracy_score(y_true_top_10_negative, binary_predictions[top_10_negative_confidence_indices]))
# Assuming binary classification, if needed, calculate and print additional metrics


# # Step 1: Rank predictions and get indices for sorting them in descending order
# sorted_indices = np.argsort(-final_predictions)  # Note the minus sign for descending sort
#
# # Step 2: Select top 10 percent
# top_10_percent_threshold = int(len(final_predictions) * 0.1)
# top_10_positive_indices = sorted_indices[:top_10_percent_threshold]
#
# # Step 3: Filter true labels and predictions based on top 10 percent indices
# y_true_top_10_positive = y_true_combined[top_10_positive_indices]
# predictions_top_10_positive = final_predictions[top_10_positive_indices]
#
# # Step 4: Convert predictions to binary based on a threshold (e.g., 0.5)
# binary_predictions_top_10_positive = (predictions_top_10_positive >= 0.5).astype(int)
#
# # Step 5: Evaluate performance on the filtered set
# roc_auc_top_10_positive = roc_auc_score(y_true_top_10_positive, predictions_top_10_positive)
# accuracy_top_10_positive = accuracy_score(y_true_top_10_positive, binary_predictions_top_10_positive)
# conf_matrix_top_10_positive = confusion_matrix(y_true_top_10_positive, binary_predictions_top_10_positive)
# class_report_top_10_positive = classification_report(y_true_top_10_positive, binary_predictions_top_10_positive)
#
# # Display results for the top 10% positive predictions
# print(f"Top 10% Positive Predictions ROC-AUC Score: {roc_auc_top_10_positive}")
# print(f"Top 10% Positive Predictions Accuracy: {accuracy_top_10_positive}")
# print("Top 10% Positive Predictions Confusion Matrix:")
# print(conf_matrix_top_10_positive)
# print("Top 10% Positive Predictions Classification Report:")
# print(class_report_top_10_positive)