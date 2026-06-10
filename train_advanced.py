import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re

print("Loading Advanced training data...")
df = pd.read_csv('data/train_features_advanced.csv')
mapping = pd.read_csv('data/user_mapping_train.csv')

# Join to get user_id for GroupKFold
df = df.merge(mapping, on='file_id')

# LightGBM requires column names to be perfectly clean
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

# Separate features and labels
X = df.drop(columns=['label', 'file_id', 'user_id'])
y = df['label']
groups = df['user_id']

print(f"Data loaded! We have {len(X)} examples, {len(X.columns)} features, and {groups.nunique()} unique users.")

# Setup the Advanced Model: LightGBM
params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'verbosity': -1
}

# GroupKFold Validation: Hiding entire users to simulate Kaggle Private LB
gkf = GroupKFold(n_splits=5)
accuracies = []
f1_scores = []

print("\nStarting GroupKFold Cross-Validation (Hiding entire users)...")

for fold, (train_index, val_index) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    
    accuracies.append(acc)
    f1_scores.append(f1)
    print(f"Fold {fold} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

print("\n=== FINAL ADVANCED RESULTS ===")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Macro F1: {np.mean(f1_scores):.4f}")

# --- Generate Submission File ---
print("\nTraining final Advanced Model on ALL data for Kaggle submission...")
model = lgb.LGBMClassifier(**params)
model.fit(X, y)

print("Loading Kaggle test features...")
test_df = pd.read_csv('data/test_features_advanced.csv')

# Clean test columns identically to training features
test_df = test_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

# Ensure test features align exactly in order and identity with training features
X_kaggle = test_df[X.columns]

print(f"Test data loaded! Predictors shape: {X_kaggle.shape}")

print("Making Kaggle predictions...")
kaggle_preds = model.predict(X_kaggle)

submission = pd.DataFrame({
    'Id': test_df['file_id'],
    'Label': kaggle_preds
})

submission.to_csv('data/submission_advanced.csv', index=False)
print("\nSuccess! Saved to: data/submission_advanced.csv")

