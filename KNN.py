from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle
import json



df = pd.read_csv('preprocessed_metadata_pakistanterrorattacks.csv')
df.head()
# Count how many times Group is empty, 'Unknown', or 'None'
pseudo_null_count = df['Group'].isin(['', 'Unknown', 'None']).sum()
print(f"Count of pseudo-null values in 'Group' column: {pseudo_null_count}")
df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
df.replace('None', pd.NA, inplace=True)
df.fillna('Unknown', inplace=True)
df = df.drop(columns=['Country', 'Day', 'city', 'latitude', 'longitude', 'Target'])
attack_mapping = {
    'Armed Assault': 'Armed',
    'Assassination': 'Armed',
    'Unarmed Assault': 'Armed',
    
    'Bombing/Explosion': 'Bombing/Explosion',
    
    'Facility/Infrastructure Attack': 'Unknown/Other',
    
    'Hijacking': 'Hijacking/Hostage',
    'Hostage Taking (Kidnapping)': 'Hijacking/Hostage',
    'Hostage Taking (Barricade Incident)': 'Hijacking/Hostage',
    
    'Unknown': 'Unknown/Other'
}

df['AttackType'] = df['AttackType'].replace(attack_mapping)


# Encode categorical features
X = df[['Year', 'Month', 'Province', 'Weapon_type', 'Group']]
y = df['AttackType']
X = pd.get_dummies(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
y_test_pred_knn = knn.predict(X_test)
test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
print(f"KNN Test Accuracy: {test_accuracy_knn:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


# Create folder if not exists
os.makedirs('models', exist_ok=True)

# Save KNN model
with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
print("KNN model saved at 'models/knn_model.pkl'")

# Save input feature columns
with open('models/knn_model_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("KNN model input columns saved at 'models/knn_model_columns.pkl'")


y_train_pred_knn = knn.predict(X_train)
train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
test_accuracy_knn = accuracy_score(y_test, y_pred)

print(f"KNN Train Accuracy: {train_accuracy_knn:.4f}")
print(f"KNN Test Accuracy: {test_accuracy_knn:.4f}")
accuracies = {}

# If the JSON file exists and has previous DT accuracies, load them first
if os.path.exists('models/accuracies.json'):
    with open('models/accuracies.json') as f:
        accuracies = json.load(f)

# Update with KNN accuracies
accuracies['knn'] = {
    "train": train_accuracy_knn,
    "test": test_accuracy_knn
}

# Save updated accuracies back to JSON
with open('models/accuracies.json', 'w') as f:
    json.dump(accuracies, f)

print("Accuracies updated with KNN results in models/accuracies.json")