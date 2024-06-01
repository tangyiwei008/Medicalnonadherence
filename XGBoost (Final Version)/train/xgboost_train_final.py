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
from scipy.stats import gmean
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.metrics import recall_score, precision_score, make_scorer


def recall_precision_weighted(y_true, y_pred, recall_weight=0.8, precision_weight=0.2):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return recall_weight * recall + precision_weight * precision


# Now create a scorer that can be used with cross-validation and model scoring
weighted_scorer = make_scorer(recall_precision_weighted)


def preprocess_data(data_filepath, labels_filepath, features_filepath):
    # Load the dataset
    df = pd.read_csv(data_filepath, low_memory=False)
    df2 = pd.read_csv(labels_filepath, low_memory=False)

    df = pd.merge(df, df2[['CLAB_ID','Non_adhere_last4']], on='CLAB_ID', how='left')

    df.drop(columns=['Week_non_adhere', 'CLAB_ID', 'days_since_claim', 'Non_adhere'], inplace=True)

    race_data = df['sdoh_race_recode_WHITE'].copy()

    # Load the top 200 features for feature selection
    top_200 = pd.read_csv(features_filepath)
    top_features = list(set(top_200['Feature'].values[:31]))

    y = df['Non_adhere_last4']
    X = df.loc[:, df.columns != 'Non_adhere_last4']
    X = X[top_features]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Imputing missing values with mean for both training and testing to prevent data leakage
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    race_test = race_data[X_test.index]

    # Apply SMOTE
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, race_test


def train_and_evaluate(X, y, X_test, y_test, scoring=weighted_scorer, i=0):
    # Calculate the ratio of negative to positive samples to set class weights
    class_weight = (y == 0).sum() / (y == 1).sum()

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
            'scale_pos_weight': class_weight,
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
                 'max_depth': (3, 5),
                 'gamma': (1, 6),
                 'min_child_weight': (5, 13),
                 'subsample': (0.5, 1.0),
                 'colsample_bytree': (0.4, 0.7),
                 'learning_rate': (0.01, 0.15),
                 'alpha': (0, 15),
                 'reg_lambda': (3, 15)},
        # pbounds={'n_estimators': (50, 300),
        #          'max_depth': (3, 10),
        #          'gamma': (0, 5),
        #          'min_child_weight': (1, 6),
        #          'subsample': (0.5, 1.0),
        #          'colsample_bytree': (0.5, 1.0),
        #          'learning_rate': (0.01, 0.3),
        #          'alpha': (0, 1),
        #          'reg_lambda': (1, 4)},
        # pbounds={'n_estimators': (50, 300),
        #          'max_depth': (3, 7),
        #          'gamma': (1, 5),
        #          'min_child_weight': (3, 10),
        #          'subsample': (0.5, 0.8),
        #          'colsample_bytree': (0.3, 0.7),
        #          'learning_rate': (0.01, 0.2),
        #          'alpha': (1, 10),
        #          'reg_lambda': (2, 10)},
        random_state=42,
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

    # Save the trained model using pickle
    with open(f'xgb_model_final.pkl', 'wb') as file:
        pickle.dump(optimized_xgb, file)

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
    plt.savefig(f'waterfall_final.png', bbox_inches='tight')

    # Clear the current figure after saving to avoid duplication in later plots
    plt.clf()


def aggregate_shap_analysis(models, X_data_list):

    # Initialize list to hold SHAP values from all models
    all_shap_values = []

    # Iterate over each model and its corresponding dataset
    for model, X in zip(models, X_data_list):
        # Initialize SHAP explainer and calculate SHAP values for the current model
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        # Append the calculated SHAP values to the list
        all_shap_values.append(shap_values.values)

    # Calculate the average SHAP values across all models
    agg_shap_values = np.max(np.array(all_shap_values), axis=0)

    return agg_shap_values


# Function to evaluate predictions based on race groups
def evaluate_combined_predictions_race(combined_predictions, y_true_combined, race_data):
    # Mask for white and non-white groups
    white_mask = (race_data == 1)
    non_white_mask = (race_data == 0)

    # Combined accuracy and ROC AUC for white group
    accuracy_white = accuracy_score(y_true_combined[white_mask], combined_predictions[white_mask] >= 0.5)
    roc_auc_white = roc_auc_score(y_true_combined[white_mask], combined_predictions[white_mask])
    fpr_white, tpr_white, _ = roc_curve(y_true_combined[white_mask], combined_predictions[white_mask])

    # Combined accuracy and ROC AUC for non-white group
    accuracy_non_white = accuracy_score(y_true_combined[non_white_mask], combined_predictions[non_white_mask] >= 0.5)
    roc_auc_non_white = roc_auc_score(y_true_combined[non_white_mask], combined_predictions[non_white_mask])
    fpr_non_white, tpr_non_white, _ = roc_curve(y_true_combined[non_white_mask], combined_predictions[non_white_mask])

    # Print the accuracy and ROC AUC results
    print(f'White Members - Accuracy: {accuracy_white}, ROC AUC: {roc_auc_white}')
    print(f'Non-White Members - Accuracy: {accuracy_non_white}, ROC AUC: {roc_auc_non_white}')

    # Plotting the ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_white, tpr_white, label=f'White ROC curve (area = {roc_auc_white:.2f})', color='blue', lw=2)
    plt.plot(fpr_non_white, tpr_non_white, label=f'Non-White ROC curve (area = {roc_auc_non_white:.2f})', color='red', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.show()


def plot_combined_shap_summary(models, X_data_list, feature_names):
    # Aggregate SHAP values
    agg_shap_values = aggregate_shap_analysis(models, X_data_list)

    plt.figure(figsize=(16, 10))
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=12)

    shap.summary_plot(agg_shap_values, features=X_data_list[-1], feature_names=feature_names, max_display=15, show=False)

    plt.subplots_adjust(left=0.5)
    plt.yticks(rotation=0)
    plt.savefig('shap_summary_plot_final.png', bbox_inches='tight')
    plt.show()


def display_top_10_percent_predictions(predictions, true_labels):
    binary_predictions = (predictions >= 0.5).astype(int)
    predictions_df = pd.DataFrame({
        'TrueLabel': true_labels,
        'PredictedProbability': predictions,
        'BinaryPrediction': binary_predictions
    })

    positive_predictions_df = predictions_df[predictions_df['BinaryPrediction'] == 1]
    sorted_positive_predictions_df = positive_predictions_df.sort_values(by='PredictedProbability', ascending=False)

    top_10_percent_cutoff = int(len(sorted_positive_predictions_df) * 0.1)
    top_10_positive_predictions_df = sorted_positive_predictions_df.head(top_10_percent_cutoff)

    y_true_top_10_positive = top_10_positive_predictions_df['TrueLabel']
    predictions_top_10_positive = top_10_positive_predictions_df['PredictedProbability']

    accuracy_top_10_positive = accuracy_score(y_true_top_10_positive, top_10_positive_predictions_df['BinaryPrediction'])
    conf_matrix_top_10_positive = confusion_matrix(y_true_top_10_positive, top_10_positive_predictions_df['BinaryPrediction'])
    class_report_top_10_positive = classification_report(y_true_top_10_positive, top_10_positive_predictions_df['BinaryPrediction'])

    # Check if both classes are present for ROC AUC calculation
    if len(np.unique(y_true_top_10_positive)) > 1:
        roc_auc_top_10_positive = roc_auc_score(y_true_top_10_positive, predictions_top_10_positive)
        print(f"Top 10% Positive Predictions ROC-AUC Score: {roc_auc_top_10_positive}")
    else:
        roc_auc_top_10_positive = np.nan  # or handle as you see fit
        print("Top 10% Positive Predictions ROC-AUC Score: Not defined (only one class present)")

    print(f"Top 10% Positive Predictions Accuracy: {accuracy_top_10_positive}")
    print("Top 10% Positive Predictions Confusion Matrix:\n", conf_matrix_top_10_positive)
    print("Top 10% Positive Predictions Classification Report:\n", class_report_top_10_positive)




# Filepaths for the weekly datasets
filepaths = [
    './merged_final.csv'
]
labels_filepath = './member_labels.csv'

features_filepath = 'feature_importance_full_final.csv'

# Initialize a dictionary to store models for each week
models = []
X_data_list = []

# Initialize lists to store weekly predictions for each subgroup
weekly_predictions_white = []
weekly_predictions_non_white = []

# Other initializations (if not already done)
weekly_feature_importance = {}
week_predictions = {}
model_metrics = {}
true_labels_weekly = {}
true_labels_white = []
true_labels_non_white = []

for i, data_filepath in enumerate(filepaths, start=1):
    print(f"Processing dataset...")

    # Preprocess the data, ensuring it returns both X and y splits
    X_train, X_test, y_train, y_test, race_data = preprocess_data(data_filepath, labels_filepath, features_filepath)

    # Reset the index of X_test and y_test if necessary to ensure alignment
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Run the train and evaluate function
    optimized_model, y_pred_proba, accuracy, roc_auc, train_sizes, train_scores, test_scores = train_and_evaluate(X_train, y_train, X_test, y_test, scoring='accuracy', i=i)

    # Save the trained model for later SHAP analysis
    models.append(optimized_model)

    # Save X_test data
    X_data_list.append(X_test)

    # Capture and save feature importance for this week
    feature_importance = optimized_model.feature_importances_
    weekly_feature_importance[f'week_{i}'] = feature_importance

    # Save the feature importance for this week as a CSV file
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    importance_df.to_csv(f'feature_importance_final.csv', index=False)

    # Store data
    week_predictions[f'week_{i}'] = y_pred_proba
    model_metrics[f'week_{i}'] = (train_sizes, train_scores, test_scores)
    true_labels_weekly[f'week_{i}'] = y_test  # Collect true labels for aggregation

    # Display results
    print(f"Accuracy of the model: {accuracy}")
    print(f"ROC-AUC Score of the model: {roc_auc}")

    # Print confusion matrix
    print(f"Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
    print(conf_matrix)

    # Print classification report
    print(f"Classification Report:")
    class_report = classification_report(y_test, (y_pred_proba > 0.5).astype(int))
    print(class_report)

    # Plot the learning curve for the current week's model
    plot_learning_curve(optimized_model, f"Learning Curve", X_train, y_train, scoring='accuracy')
    plot_learning_curve(optimized_model, f"Learning Curve", X_train, y_train, scoring='neg_log_loss')

    # Plot the ROC curve for the current week's model
    plot_roc_curve(y_test, y_pred_proba, title=f'ROC Curve')

    # Plot SHAP Waterfall
    # You can adjust the instance index as needed
    plot_shap_waterfall(optimized_model, X_test, instance_index=0, week=i)


# Assuming X_train.columns and weekly_feature_importance are a dictionary where keys are week numbers
feature_names = X_train.columns
num_features = len(feature_names)
num_weeks = len(weekly_feature_importance)

# After processing all weeks, aggregate true labels to form y_true_combined
# This example assumes y_test for each week is a binary array indicating the event occurrence
y_true_combined = np.logical_or.reduce([true_labels_weekly[f'week_{i}'] for i in range(1, 2)])

# Process final_predictions
combined_predictions = np.max(np.column_stack(list(week_predictions.values())), axis=1)
final_binary_predictions = (combined_predictions >= 0.5).astype(int)

# Evaluate combined model performance
roc_auc_combined = roc_auc_score(y_true_combined, combined_predictions)
accuracy_combined = accuracy_score(y_true_combined, final_binary_predictions)
conf_matrix_combined = confusion_matrix(y_true_combined, final_binary_predictions)
class_report_combined = classification_report(y_true_combined, final_binary_predictions)

# Display combined model results
print(f"ROC-AUC Score: {roc_auc_combined}")
print(f"Accuracy: {accuracy_combined}")
print("Confusion Matrix:")
print(conf_matrix_combined)
print("Classification Report:")
print(class_report_combined)

# Combined race evaluation
evaluate_combined_predictions_race(combined_predictions, y_true_combined, race_data)

# Plot ROC curve for combined model predictions
plot_roc_curve(y_true_combined, combined_predictions, title='ROC Curve')

# Plot combined SHAP plot
plot_combined_shap_summary(models, X_data_list, feature_names)

# Generate Predictive Results for top 10% of Positive Predictions
display_top_10_percent_predictions(combined_predictions, y_true_combined)
