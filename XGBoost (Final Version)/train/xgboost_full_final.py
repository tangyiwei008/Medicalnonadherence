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
from imblearn.over_sampling import SMOTE
from scipy.stats import gmean
from sklearn.metrics import recall_score, precision_score, make_scorer


def recall_precision_weighted(y_true, y_pred, recall_weight=0.8, precision_weight=0.2):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return recall_weight * recall + precision_weight * precision


# Now create a scorer that can be used with cross-validation and model scoring
weighted_scorer = make_scorer(recall_precision_weighted)


def preprocess_data(filepath, filepath2):
    # Load the dataset
    df = pd.read_csv(filepath, low_memory=False)
    df2 = pd.read_csv(filepath2, low_memory=False)

    df = pd.merge(df, df2[['CLAB_ID','Non_adhere_last4']], on='CLAB_ID', how='left')

    df.drop(columns=['Week_non_adhere', 'CLAB_ID', 'days_since_claim', 'Non_adhere'], inplace=True)

    y = df['Non_adhere_last4']
    X = df.loc[:, df.columns != 'Non_adhere_last4']

    # Imputing missing values
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = pd.DataFrame(fill_NaN.fit_transform(X), columns=X.columns)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X, y, X_test, y_test, IDs_test, week_num_adhere_test, i, scoring=weighted_scorer):
    # Calculate the ratio of negative to positive samples to set class weights
    class_weight = (y == 0).sum() / (y == 1).sum()

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
        # Other tuning settings for BO hyperparameter tuning.
        # Pick which one is best and comment out the others
        # pbounds={'n_estimators': (50, 300),
        #          'max_depth': (3, 5),
        #          'gamma': (1, 6),
        #          'min_child_weight': (5, 13),
        #          'subsample': (0.5, 1.0),
        #          'colsample_bytree': (0.4, 0.7),
        #          'learning_rate': (0.01, 0.15),
        #          'alpha': (0, 15),
        #          'reg_lambda': (3, 15)},
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
                 'max_depth': (3, 7),
                 'gamma': (1, 5),
                 'min_child_weight': (3, 10),
                 'subsample': (0.5, 0.8),
                 'colsample_bytree': (0.3, 0.7),
                 'learning_rate': (0.01, 0.2),
                 'alpha': (1, 10),
                 'reg_lambda': (2, 10)},
        random_state=42,
    )

    # Running the optimization process
    xgb_bo.maximize(n_iter=2, init_points=3)

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

    # Create DataFrame with CLAB_ID, Non_adhere, Week_non_adhere, and Prediction
    predictions_df = pd.DataFrame({
        'CLAB_ID': IDs_test,
        'Non_adhere': y_test,
        'Week_non_adhere': week_num_adhere_test,
        'Prediction': y_pred_proba
    })

    # Save predictions to a CSV file
    # predictions_df.to_csv(f'predictions_xgb_full_final.csv', index=False)

    # Return the model, predictions, evaluation metrics, learning curve data, and predictions DataFrame
    return optimized_xgb, y_pred_proba, accuracy, roc_auc, train_sizes, train_scores, test_scores, predictions_df


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
    shap.plots.waterfall(shap_values[instance_index], max_display=16, show=False)

    # Save the figure
    plt.savefig(f'waterfall_{week}.png', bbox_inches='tight')

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


# Filepaths for the weekly datasets
filepaths = [
    './merged_final.csv'
]
filepath2 = './member_labels.csv'

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

for i, filepath in enumerate(filepaths, start=1):
    print(f"Processing dataset...")

    # Preprocess the data, ensuring it returns both X and y splits
    X_train, X_test, y_train, y_test = preprocess_data(filepath, filepath2)

    # Reset the index of X_test and y_test if necessary to ensure alignment
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    df = pd.read_csv(filepath, low_memory=False)
    IDs = df['CLAB_ID']
    week_num_adhere = df['Week_non_adhere']

    # Run the train and evaluate function
    # Make sure to pass IDs and Week_non_adhere to train_and_evaluate function
    optimized_model, y_pred_proba, accuracy, roc_auc, train_sizes, train_scores, test_scores, predictions_df = train_and_evaluate(
        X_train, y_train, X_test, y_test, IDs[X_test.index], week_num_adhere[X_test.index], i, scoring='accuracy'
    )

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

    importance_df.to_csv(f'feature_importance_full_final.csv', index=False)
