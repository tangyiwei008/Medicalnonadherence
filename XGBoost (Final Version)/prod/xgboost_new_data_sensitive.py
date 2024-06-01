import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import pickle


def preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath, low_memory=False)

    # Basic preprocessing steps, assume Non_adhere column might not be present in production
    if 'Non_adhere' in df.columns:
        df = df.drop('Non_adhere', axis=1)

    # Keep CLAB_ID if it's meant to identify members across predictions
    IDs = df['CLAB_ID'] if 'CLAB_ID' in df.columns else None
    IDs = IDs.astype(int)
    df.drop(['Week_non_adhere', 'CLAB_ID'], axis=1, inplace=True, errors='ignore')

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return X, IDs


def load_model_and_predict(model_file, X):
    # Load the pre-trained XGBoost model
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    cols_when_model_builds = model.get_booster().feature_names
    X = X[cols_when_model_builds]

    # Predict using the loaded model
    y_pred_proba = model.predict_proba(X)[:, 1]  # probability of class 1
    return y_pred_proba


def main():
    filepath = './merged_new_data.csv'
    X, IDs = preprocess_data(filepath)
    all_predictions = []

    # Iterate over each model week
    # for i in range(1, 5):
    #     model_file = f'xgb_temporal_model_w{i}.pkl'
    #     predictions = load_model_and_predict(model_file, X)
    #     all_predictions.append(predictions)

    model_file = f'xgb_model_final_sensitive.pkl'
    predictions = load_model_and_predict(model_file, X)
    all_predictions.append(predictions)

    # Take the maximum prediction for each member across weeks
    max_predictions = np.max(all_predictions, axis=0)

    # Create a DataFrame for the final predictions
    final_predictions_df = pd.DataFrame({
        'CLAB_ID': IDs,
        'Max_Prediction': max_predictions
    })

    # print(final_predictions_df[final_predictions_df['Max_Prediction'] < 0.5])

    # Save the final predictions to a CSV file
    final_predictions_df.to_csv('new_predictions_final_sensitive.csv', index=False)
    print("Final predictions saved to 'new_predictions_final_sensitive.csv'.")

if __name__ == "__main__":
    main()
