"""
Ablation study for Assignment 3 report (Q4).
Run AFTER data_processing.py and create_user_mapping.py have completed.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score
import re

df = pd.read_csv('data/train_features_advanced.csv')
mapping = pd.read_csv('data/user_mapping_train.csv')
df = df.merge(mapping, on='file_id')
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
y = df['label']
groups = df['user_id']
gkf = GroupKFold(n_splits=5)
params = dict(n_estimators=300, learning_rate=0.05,
              class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1)

all_cols = [c for c in df.columns if c not in ['label', 'file_id', 'user_id']]

def run(cols, label):
    X = df[cols].fillna(0)
    f1s = []
    for tr, va in gkf.split(X, y, groups):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        f1s.append(f1_score(y.iloc[va], m.predict(X.iloc[va]), average='macro'))
    mean, std = np.mean(f1s), np.std(f1s)
    print(f"  {label:60s}  F1={mean:.4f}  std={std:.4f}")
    return mean

overall_time = [c for c in all_cols if c.startswith('overall_') and
                not any(s in c for s in ['fft','jerk','corr','sma'])]
overall_fft  = [c for c in all_cols if c.startswith('overall_') and 'fft' in c]
overall_jerk = [c for c in all_cols if c.startswith('overall_') and 'jerk' in c]
overall_corr = [c for c in all_cols if c.startswith('overall_') and ('corr' in c or 'sma' in c)]
window_cols  = [c for c in all_cols if any(c.startswith(f'w{i}_') for i in range(1,4))]

print("=== Ablation Study ===\n")
r1 = run(overall_time,                                          "A) Overall time-domain stats only")
r2 = run(overall_time + overall_fft,                            "B) A + FFT (frequency domain)")
r3 = run(overall_time + overall_fft + overall_jerk,             "C) B + jerk signals")
r4 = run(overall_time + overall_fft + overall_jerk + overall_corr, "D) C + cross-axis corr + SMA")
r5 = run(all_cols,                                              "E) D + 3 sub-windows (FULL)")

print("\n=== GroupKFold vs random split ===")
X_full = df[all_cols].fillna(0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1s_rand = []
for tr, va in skf.split(X_full, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(X_full.iloc[tr], y.iloc[tr])
    f1s_rand.append(f1_score(y.iloc[va], m.predict(X_full.iloc[va]), average='macro'))
print(f"  {'Random KFold (leaks users — optimistic)':60s}  F1={np.mean(f1s_rand):.4f}")
print(f"  {'GroupKFold (hides users — realistic)':60s}  F1={r5:.4f}")
print("\nDone!")