import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import json


# Load data
df = pd.read_csv('preprocessed_metadata_pakistanterrorattacks.csv')
df.head()

# Handle pseudo-null values in 'Group'
pseudo_null_count = df['Group'].isin(['', 'Unknown', 'None']).sum()
print(f"Count of pseudo-null values in 'Group' column: {pseudo_null_count}")

df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
df.replace('None', pd.NA, inplace=True)
df.fillna('Unknown', inplace=True)

# Drop unwanted columns
df = df.drop(columns=['Country', 'Day', 'city', 'latitude', 'longitude', 'Target'])

# Map AttackType to fewer categories
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

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(df['AttackType'])  # Labels 0 to 3

# Prepare features with one-hot encoding
X = df[['Year', 'Month', 'Province', 'Weapon_type', 'Group']]
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Class weights (optional, for imbalanced classes)
class_weights = {
    0: 1829 / 1145,  # Bombing/Explosion
    1: 1829 / 566,   # Armed
    2: 1829 / 77,    # Hijacking/Hostage
    3: 1829 / 41     # Unknown/Other
}

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight=class_weights,
    random_state=42
)

# Train model
clf.fit(X_train, y_train)

# Predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Evaluation
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))

# Create models directory if doesn't exist
os.makedirs('models', exist_ok=True)

# Save Decision Tree model
model_path = 'models/decision_tree_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"Decision Tree model saved at {model_path}")

# Save Label Encoder
encoder_path = 'models/label_encoder.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(le, f)
print(f"Label Encoder saved at {encoder_path}")

# Save one-hot encoded columns list
columns_path = 'models/dt_columns.pkl'
with open(columns_path, 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print(f"Training columns saved at {columns_path}")



accuracies = {}

if os.path.exists('models/accuracies.json'):
    with open('models/accuracies.json') as f:
        accuracies = json.load(f)

accuracies['decision_tree'] = {
    "train": accuracy_score(y_train, y_train_pred),
    "test": accuracy_score(y_test, y_test_pred)
}

with open('models/accuracies.json', 'w') as f:
    json.dump(accuracies, f)

print("Accuracies updated with Decision Tree results in models/accuracies.json")