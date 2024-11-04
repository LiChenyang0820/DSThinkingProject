from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import numpy as np


#%%
# Step 1: Load the dataset
file_path = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DSThinkingProject\SVM\S1Data.csv'
data = pd.read_csv(file_path)

#%%
# Step 2: Separate features and target
X = data.drop(['Event','TIME'], axis=1)  # Features
y = data['Event']  # Target (Event column)
feature_names = X.columns

# Step 3: Initialize results dictionary for multiple runs
results = {'accuracy': [], 'f1_score': [], 'roc_auc': [], 'mcc': []}
feature_rankings = []

#%%
# Step 4: Run the experiment 100 times
for i in range(100):
    print(f'\nExperiment {i+1}/100')
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Standardize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate the Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Make predictions
    y_prob = nb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Collect the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)

    results['accuracy'].append(accuracy)
    results['f1_score'].append(f1)
    results['roc_auc'].append(roc_auc)
    results['mcc'].append(mcc)

    # Calculate feature importance using mutual information
    mi = mutual_info_classif(X_train, y_train, discrete_features='auto')
    feature_ranking = sorted(zip(feature_names, mi), key=lambda x: x[1], reverse=True)
    feature_rankings.append([feature for feature, _ in feature_ranking])

#%%
# Step 5: Aggregate the results
final_results = {
    'Accuracy': np.mean(results['accuracy']),
    'F1 score': np.mean(results['f1_score']),
    'ROC AUC': np.mean(results['roc_auc']),
    'MCC': np.mean(results['mcc'])
}

# Visualize the final evaluation metrics
import matplotlib.pyplot as plt

metrics = ['Accuracy', 'F1 score', 'ROC AUC', 'MCC']
values = [final_results[metric] for metric in metrics]
colors = ['blue', 'green', 'red', 'cyan']


plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=colors)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Averaged Evaluation Metrics over 100 Runs (Naive Bayes)')
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.6f}', ha='center')
plt.tight_layout()
plt.show()

# Print the results
print("\nFinal Evaluation Metrics (Averaged over 100 Runs):")
for metric, value in final_results.items():
    print(f"{metric.capitalize()}: {value:.6f}")

#%%
# Step 6: Calculate average feature importance
feature_rank_count = defaultdict(int)
for ranking in feature_rankings:
    for rank, feature in enumerate(ranking):
        feature_rank_count[feature] += rank + 1

average_ranking = sorted(feature_rank_count.items(), key=lambda x: x[1] / len(feature_rankings))

print("\nAverage Feature Ranking:")
for rank, (feature, total_rank) in enumerate(average_ranking, start=1):
    print(f"{rank}. Feature: {feature}, Average Rank: {total_rank / len(feature_rankings):.2f}")

# Visualize the average feature ranking
features, avg_ranks = zip(*[(feature, total_rank / len(feature_rankings)) for feature, total_rank in average_ranking])
plt.figure(figsize=(10, 6))
plt.barh(features, avg_ranks, color='skyblue', height=0.5)
plt.xlabel('Average Rank')
plt.title('Average Feature Ranking Over 100 Runs (Naive Bayes)')
plt.gca().invert_yaxis()
for i, v in enumerate(avg_ranks):
    plt.text(v + 0.01, i, f'{v:.2f}', va='center')
plt.tight_layout()
plt.show()