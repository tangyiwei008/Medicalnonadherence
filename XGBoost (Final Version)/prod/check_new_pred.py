import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Read data from files
df = pd.read_csv('new_predictions_final.csv')
labels = pd.read_csv('./member_labels.csv')

# Merge the prediction and label dataframes on the common column 'CLAB_ID'
df = pd.merge(df, labels[['CLAB_ID', 'Non_adhere_last4', 'Non_adhere_flip']], on='CLAB_ID', how='left')

# Convert probabilities to binary predictions (1 for probabilities > 0.5, otherwise 0)
df['pred'] = (df['Max_Prediction'] > 0.5).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(df['Non_adhere_last4'], df['pred'])
print("Confusion Matrix:")
print(cm)

# Generate and print classification report
report = classification_report(df['Non_adhere_last4'], df['pred'])
print("Classification Report:")
print(report)

# Top 10% performance evaluation
# Sort by 'Max_Prediction' in descending order to get the top 10% highest probabilities
top_10_percent = df.sort_values(by='Max_Prediction', ascending=False).head(int(0.1 * len(df)))

# Calculate metrics for top 10%
top_10_cm = confusion_matrix(top_10_percent['Non_adhere_last4'], top_10_percent['pred'])
top_10_report = classification_report(top_10_percent['Non_adhere_last4'], top_10_percent['pred'])
top_10_accuracy = accuracy_score(top_10_percent['Non_adhere_last4'], top_10_percent['pred'])

print("\nTop 10% Predictions Analysis")
print("Confusion Matrix for Top 10% Predictions:")
print(top_10_cm)
print("Classification Report for Top 10% Predictions:")
print(top_10_report)
print(f"Accuracy for Top 10% Predictions: {top_10_accuracy:.2f}")

# Results for "flip"
# Filter for the 'flip' cases and evaluate
df_flip = df[df['Non_adhere_flip'] == 1]
flip_accuracy = accuracy_score(df_flip['Non_adhere_last4'], df_flip['pred'])
# flip_auc = roc_auc_score(df_flip['Non_adhere_last4'], df_flip['pred'])

print('Results for those who remained adherent then went non-adherent:')
print(f"Flip Case Accuracy: {flip_accuracy}")
print("Confusion Matrix:")
print(confusion_matrix(df_flip['Non_adhere_last4'], df_flip['pred']))
print("Classification Report:")
print(classification_report(df_flip['Non_adhere_last4'], df_flip['pred']))
