import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import shap
pd.set_option('display.min_rows', 20)


def preprocess_data(filepath, features_filepath='feature_importance_final.csv'):
    # Load the dataset
    df = pd.read_csv(filepath, low_memory=False)

    # Basic preprocessing steps
    df = df[~df['Non_adhere'].isna()]
    df = df[df['Week_non_adhere'] > 5]  # Ensuring an adequate period of adherence
    df.drop(columns=['Week_non_adhere', 'CLAB_ID'], inplace=True)

    y = df['Non_adhere']
    X = df.loc[:, df.columns != 'Non_adhere']

    # Drop county column (for now)
    X.drop(columns='CNTY_NM', inplace=True)

    # Load the top 200 features
    top_200 = pd.read_csv(features_filepath)
    top_200 = top_200[top_200['Max_Importance'] > 0]
    top_200 = top_200.iloc[:201, :]
    cols_to_keep = list(top_200['Feature'].values)

    # Handling specific bugged columns
    bugged_cols = ['CLMDRUGPLN_CD_D093900R_w1', 'CLMDRUGPLN_CD_D093600R_w1', 'DRUGACCT_CD_MNH5959014_w1',
                   'CLMDRUGPLN_CD_D093900R_w5', 'DRUGACCT_CD_MNH5959009_w5']
    for b in bugged_cols:
        if b in cols_to_keep:
            cols_to_keep.remove(b)

    X = X[cols_to_keep]

    # Dropping columns directly related to the outcome variable
    off_limits = ['cas_nums_172', 'cas_nums_10119', 'cas_nums_39', 'cas_nums_49', 'cas_nums_17', 'cas_nums_23',
                  'cas_nums_24', 'cas_nums_25', 'cas_nums_26', 'cas_nums_29', 'cas_nums_50', 'cas_nums_51',
                  'cas_nums_40',
                  'cas_nums_41', 'cas_nums_48', 'cas_nums_120']
    cols_to_drop2 = [col for col in cols_to_keep if any(bad_root in col for bad_root in off_limits)]

    X.drop(columns=cols_to_drop2, inplace=True)

    # Dropping race-sensitive features
    race_sensitive = ['RACE', 'ETHNICITY', 'COUNTRY']
    cols_to_drop3 = [col for col in cols_to_keep if any(bad_root in col for bad_root in race_sensitive)]
    X.drop(columns=cols_to_drop3, inplace=True)

    # Imputing missing values
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = pd.DataFrame(fill_NaN.fit_transform(X), columns=X.columns)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X, y, X_test, y_test, scoring='accuracy'):
    # Define the hyperparameter optimization function
    def bo_tune_xgb(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree, learning_rate, alpha, reg_lambda):
        params = {
            'n_estimators': int(round(n_estimators)),
            'max_depth': int(round(max_depth)),
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'lambda': reg_lambda,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        xg_clf = xgb.XGBClassifier(**params, random_state=42)
        cv_scores = cross_val_score(xg_clf, X, y, cv=5, scoring=scoring, n_jobs=-1)
        return cv_scores.mean()

    # Set up Bayesian Optimization
    xgb_bo = BayesianOptimization(
        f=bo_tune_xgb,
        pbounds={'n_estimators': (50, 300),
                 'max_depth': (3, 7),
                 'gamma': (1, 5),
                 'min_child_weight': (3, 10),
                 'subsample': (0.5, 0.8),
                 'colsample_bytree': (0.3, 0.7),
                 'learning_rate': (0.01, 0.2),
                 'alpha': (1, 10),
                 'reg_lambda': (2, 10)},
        random_state=1,
    )

    # Running the optimization process
    xgb_bo.maximize(n_iter=5, init_points=8)

    # Extracting the best parameters
    best_params = xgb_bo.max['params']
    best_params['n_estimators'] = int(round(best_params['n_estimators']))
    best_params['max_depth'] = int(round(best_params['max_depth']))

    # Training the XGBoost model with optimized parameters
    optimized_xgb = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    optimized_xgb.fit(X, y)

    # Capture the training and test scores for plotting learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        optimized_xgb, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring=scoring)

    # Making predictions and evaluating the model
    y_pred = optimized_xgb.predict(X_test)
    y_pred_proba = optimized_xgb.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Return the model, predictions, evaluation metrics, and learning curve data
    return optimized_xgb, y_pred_proba, accuracy, roc_auc, train_sizes, train_scores, test_scores


def plot_learning_curve(estimator, title, X, y, scoring='accuracy', ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    ylabel = "Accuracy" if scoring == 'accuracy' else "Log Loss"
    plt.ylabel(ylabel)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes, scoring=scoring)

    if scoring == 'accuracy':
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
    else:  # For 'neg_log_loss', the sign should be flipped to make it positive for easier interpretation
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)
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


def plot_roc_curve(y_test, y_pred_proba, title='ROC Curve'):
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_shap_waterfall(model, X, week, instance_index=0):
    # Initialize SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Create a new figure for the plot
    plt.figure()

    # Generate waterfall plot for the specified instance
    # Since SHAP's plot function directly uses matplotlib, the plot will be made on the current figure
    shap.plots.waterfall(shap_values[instance_index], max_display=15, show=False)

    # Save the figure
    plt.savefig(f'waterfall_{week}.png', bbox_inches='tight')

    # Clear the current figure after saving to avoid duplication in later plots
    plt.clf()


# Filepaths for the weekly datasets
filepaths = [
    'merged_model_1week.csv',
    'merged_model_2week.csv',
    'merged_model_3week.csv',
    'merged_model_4week.csv'
]

# Initialize dictionaries to store data
model_metrics = {}
week_predictions = {}
true_labels_weekly = {}  # Store true labels for each week
weekly_feature_importance = {}

for i, filepath in enumerate(filepaths, start=1):
    print(f"Processing dataset for Week {i}...")

    # Preprocess the data, ensuring it returns both X and y splits
    X_train, X_test, y_train, y_test = preprocess_data(filepath)

    # Run the train and evaluate function
    optimized_model, y_pred_proba, accuracy, roc_auc, train_sizes, train_scores, test_scores = train_and_evaluate(X_train, y_train, X_test, y_test, scoring='accuracy')

    # Capture and save feature importance for this week
    feature_importance = optimized_model.feature_importances_
    weekly_feature_importance[f'week_{i}'] = feature_importance

    # Save the feature importance for this week as a CSV file
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    importance_df.to_csv(f'feature_importance_week_{i}.csv', index=False)

    # Store data
    week_predictions[f'week_{i}'] = y_pred_proba
    model_metrics[f'week_{i}'] = (train_sizes, train_scores, test_scores)
    true_labels_weekly[f'week_{i}'] = y_test  # Collect true labels for aggregation

    # Display results
    print(f"Accuracy of the model for Week {i}: {accuracy}")
    print(f"ROC-AUC Score of the model for Week {i}: {roc_auc}")

    # Print confusion matrix
    print(f"Confusion Matrix for Week {i}:")
    conf_matrix = confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
    print(conf_matrix)

    # Print classification report
    print(f"Classification Report for Week {i}:")
    class_report = classification_report(y_test, (y_pred_proba > 0.5).astype(int))
    print(class_report)

    # Plot the learning curve for the current week's model
    plot_learning_curve(optimized_model, f"Learning Curve for Week {i}", X_train, y_train, scoring='accuracy')
    plot_learning_curve(optimized_model, f"Learning Curve for Week {i}", X_train, y_train, scoring='neg_log_loss')

    # Plot the ROC curve for the current week's model
    plot_roc_curve(y_test, y_pred_proba, title=f'ROC Curve for Week {i}')

    # Plot SHAP Waterfall
    # You can adjust the instance index as needed
    if len(X_test) > 0:  # Check if X_test is not empty
        plot_shap_waterfall(optimized_model, X_test, instance_index=0, week=i)


# Aggregate feature importance
feature_names = X_train.columns  # Ensure this matches across all weeks
num_features = len(feature_names)
num_weeks = len(weekly_feature_importance)
summed_importance = np.zeros(num_features)

for week_importance in weekly_feature_importance.values():
    summed_importance += week_importance

# Compute the average by dividing the summed importance by the number of weeks
average_importance = summed_importance / num_weeks

# Create and save the DataFrame
average_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'AverageImportance': average_importance
}).sort_values(by='AverageImportance', ascending=False)

average_importance_df.to_csv('feature_importance_week_all.csv', index=False)

# After processing all weeks, aggregate true labels to form y_true_combined
# This example assumes y_test for each week is a binary array indicating the event occurrence
y_true_combined = np.logical_or.reduce([true_labels_weekly[f'week_{i}'] for i in range(1, 5)])

# Process final_predictions as before
final_predictions = np.max(np.column_stack(list(week_predictions.values())), axis=1)
final_binary_predictions = (final_predictions >= 0.5).astype(int)

# Evaluate combined model performance
roc_auc_combined = roc_auc_score(y_true_combined, final_predictions)
accuracy_combined = accuracy_score(y_true_combined, final_binary_predictions)
conf_matrix_combined = confusion_matrix(y_true_combined, final_binary_predictions)
class_report_combined = classification_report(y_true_combined, final_binary_predictions)

# Display combined model results
print(f"Combined Model ROC-AUC Score: {roc_auc_combined}")
print(f"Combined Model Accuracy: {accuracy_combined}")
print("Combined Model Confusion Matrix:")
print(conf_matrix_combined)
print("Combined Model Classification Report:")
print(class_report_combined)

# Plot ROC curve for combined model predictions
plot_roc_curve(y_true_combined, final_predictions, title='ROC Curve for Combined Model')

# Generate Predictive Results for top 10% of Positive Predictions
# Convert predicted probabilities to binary predictions using 0.5 as the threshold
binary_predictions = (final_predictions >= 0.5).astype(int)

# Combine true labels and predictions into a DataFrame for easier manipulation
predictions_df = pd.DataFrame({
    'TrueLabel': y_true_combined,
    'PredictedProbability': final_predictions,
    'BinaryPrediction': binary_predictions
})

# Filter to only include instances where the model predicts positive (outcome = 1)
positive_predictions_df = predictions_df[predictions_df['BinaryPrediction'] == 1]

# Sort these positive predictions by their probability (confidence) in descending order
sorted_positive_predictions_df = positive_predictions_df.sort_values(by='PredictedProbability', ascending=False)

# Select the top 10% of these sorted positive predictions
top_10_percent_cutoff = int(len(sorted_positive_predictions_df) * 0.1)
top_10_positive_predictions_df = sorted_positive_predictions_df.head(top_10_percent_cutoff)

# Extract true labels and predictions from the top 10% positive predictions DataFrame
y_true_top_10_positive = top_10_positive_predictions_df['TrueLabel']
predictions_top_10_positive = top_10_positive_predictions_df['PredictedProbability']

# Calculate metrics for the top 10% positive predictions
# roc_auc_top_10_positive = roc_auc_score(y_true_top_10_positive, predictions_top_10_positive)
accuracy_top_10_positive = accuracy_score(y_true_top_10_positive, top_10_positive_predictions_df['BinaryPrediction'])
conf_matrix_top_10_positive = confusion_matrix(y_true_top_10_positive, top_10_positive_predictions_df['BinaryPrediction'])
class_report_top_10_positive = classification_report(y_true_top_10_positive, top_10_positive_predictions_df['BinaryPrediction'])

# Display results for the top 10% positive predictions
# print(f"Top 10% Positive Predictions ROC-AUC Score: {roc_auc_top_10_positive}")
print(f"Top 10% Positive Predictions Accuracy: {accuracy_top_10_positive}")
print("Top 10% Positive Predictions Confusion Matrix:\n", conf_matrix_top_10_positive)
print("Top 10% Positive Predictions Classification Report:\n", class_report_top_10_positive)

