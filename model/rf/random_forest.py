import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")
data = data.drop(columns=['time'])

# Features and target variable
X = data.drop(columns=['DEATH_EVENT'])
y = data['DEATH_EVENT']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize lists to store metrics
mcc_scores = []
f1_scores = []
accuracy_scores = []
roc_auc_scores = []

# Feature ranking statistics
feature_rank_count = defaultdict(int)
feature_rankings = []

# Perform 100 iterations of random train-test splits and model training
for _ in range(100):
    # Randomly split the dataset without a fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
    
    # Train Random Forest model with increased complexity
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Predict and calculate metrics
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]  # Used for ROC AUC calculation
    
    # Metrics based on 0.5 threshold
    mcc_scores.append(matthews_corrcoef(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    
    # ROC AUC (automatically selects the best threshold)
    roc_auc = roc_auc_score(y_test, y_prob)
    roc_auc_scores.append(roc_auc)
    
    # Feature importance ranking
    feature_importances = rf_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    ranking = [X.columns[i] for i in sorted_indices]
    
    # Accumulate ranking values
    for rank, feature in enumerate(ranking):
        feature_rank_count[feature] += rank + 1
    feature_rankings.append(ranking)

# Calculate average ranking
average_ranking = sorted(
    [(feature, total_rank / len(feature_rankings)) for feature, total_rank in feature_rank_count.items()],
    key=lambda x: x[1]
)

feature_label_mapping = {
    'serum_creatinine': 'Creatinine',
    'ejection_fraction': 'Ejection.Fraction',
    'age': 'Age',
    'creatinine_phosphokinase': 'CPK',
    'serum_sodium': 'Sodium',
    'high_blood_pressure': 'BP',
    'sex': 'Gender',
    'smoking': 'Smoking',
    'anaemia': 'Anaemia',
    'diabetes': 'Diabetes',
    'platelets': 'Platelets'
}

features, avg_ranks = zip(*[(feature, avg_rank) for feature, avg_rank in average_ranking])
mapped_labels = [feature_label_mapping[feature] for feature in features]

# Save feature ranking chart with mapped labels on y-axis
plt.figure(figsize=(10, 6))
plt.barh(mapped_labels, avg_ranks, color='blue', height=0.5)
plt.xlabel('Average Rank')
plt.title('Average Feature Ranking Over 100 Runs (Random Forest)')
plt.gca().invert_yaxis()

# Add value labels
for i, v in enumerate(avg_ranks):
    plt.text(v + 0.1, i, f'{v:.2f}', va='center')

plt.tight_layout()
plt.savefig('RF_feature_ranking.png') 
plt.close()

# Calculate average metric scores
avg_accuracy = np.mean(accuracy_scores)
avg_f1 = np.mean(f1_scores)
avg_roc_auc = np.mean(roc_auc_scores)
avg_mcc = np.mean(mcc_scores)

# Metric names and average values in the specified order
metrics = ['Accuracy', 'F1 Score', 'ROC AUC', 'MCC']
values = [avg_accuracy, avg_f1, avg_roc_auc, avg_mcc]

# Save evaluation metrics chart with specified format
plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color='skyblue')
plt.ylabel('Score')
plt.title('Average Evaluation Metrics Over 100 Runs (Random Forest)')

# Add value labels with six decimal places
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.6f}', ha='center')

plt.tight_layout()
plt.savefig('RF_evaluation_metrics.png') 
plt.close()

# Output average metric values
print(f"Average Accuracy: {avg_accuracy:.6f}")
print(f"Average F1 Score: {avg_f1:.6f}")
print(f"Average ROC AUC: {avg_roc_auc:.6f}")
print(f"Average MCC: {avg_mcc:.6f}")

# Output average feature ranking
print("\nAverage Feature Ranking:")
for rank, (feature, avg_rank) in enumerate(average_ranking, start=1):
    print(f"{rank}. Feature: {feature_label_mapping[feature]}, Average Rank: {avg_rank:.2f}")
