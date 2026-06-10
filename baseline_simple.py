"""
Baseline comparison script for Assignment 3 report (Q1 & Q2).
Run AFTER data_processing.py and create_user_mapping.py have completed.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import re

df = pd.read_csv('data/train_features_advanced.csv')
mapping = pd.read_csv('data/user_mapping_train.csv')
df = df.merge(mapping, on='file_id')
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
groups = df['user_id']
y = df['label']

# Print actual column names so we can see what we have
all_cols = [c for c in df.columns if c not in ['label', 'file_id', 'user_id']]
print(f"Total features: {len(all_cols)}")
print("Sample column names:", all_cols[:5])

gkf = GroupKFold(n_splits=5)
params = dict(n_estimators=300, learning_rate=0.05,
              class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1)

def evaluate(X, y, groups, model_fn, label):
    f1s = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        m = model_fn()
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[val_idx])
        f1s.append(f1_score(y.iloc[val_idx], preds, average='macro'))
    print(f"  {label:55s}  Macro F1 = {np.mean(f1s):.4f}")
    return np.mean(f1s)

# 1. Majority class baseline (uses any single column as dummy input)
X_dummy = df[[all_cols[0]]].fillna(0)
evaluate(X_dummy, y, groups,
         lambda: DummyClassifier(strategy='most_frequent'),
         "Majority class (dummy)")

# 2. Overall time-domain stats only (no FFT, no jerk, no corr, no sma)
overall_time = [c for c in all_cols if c.startswith('overall_') and
                not any(s in c for s in ['fft', 'jerk', 'corr', 'sma'])]
evaluate(df[overall_time].fillna(0), y, groups,
         lambda: lgb.LGBMClassifier(**params),
         "Overall time-domain only (LightGBM)")

# 3. Overall stats + FFT (no sub-windows)
overall_all = [c for c in all_cols if c.startswith('overall_')]
evaluate(df[overall_all].fillna(0), y, groups,
         lambda: lgb.LGBMClassifier(**params),
         "Overall time+freq domain (LightGBM)")

# 4. Full feature set
evaluate(df[all_cols].fillna(0), y, groups,
         lambda: lgb.LGBMClassifier(**params),
         "Full feature set — overall + sub-windows (LightGBM)")

print("\nLabel distribution in training set:")
print(df['label'].value_counts().sort_index().to_string())