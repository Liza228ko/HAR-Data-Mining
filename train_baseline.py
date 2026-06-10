import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data
print("Loading master training data...")
df = pd.read_csv('data/train_features.csv')

# 2. Separate Features (X) from the Answer/Label (y)
# We drop 'label' because it's the answer we want to predict.
# We drop 'file_id' because it's just an ID number and has nothing to do with movement.
X = df.drop(columns=['label', 'file_id'])
y = df['label']

print(f"Data loaded! We have {len(X)} examples and {len(X.columns)} features to learn from.")

# 3. Setup the Baseline Model
# Random Forest creates 100 "decision trees" and has them vote on the answer.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 4. Setup Cross-Validation
# We split the data into 5 equal pieces. We train on 4 pieces and test on the 1 piece we hid.
# 'Stratified' is crucial: it ensures every piece has the exact same ratio of Activity 0, 1, 2, etc.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nStarting Cross-Validation (Training and Testing 5 times)...")
accuracies = []

fold = 1
for train_index, test_index in skf.split(X, y):
    # Get the training data and the hidden testing data for this round
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Let the model learn from the training data!
    model.fit(X_train, y_train)
    
    # Ask the model to guess the answers for the hidden testing data
    predictions = model.predict(X_test)
    
    # Grade the model's guesses
    acc = accuracy_score(y_test, predictions)
    accuracies.append(acc)
    print(f"Round {fold} Accuracy: {acc:.4f} (or {acc * 100:.2f}%)")
    fold += 1

# 5. Final Results
average_accuracy = np.mean(accuracies)
print("\n=== FINAL BASELINE RESULTS ===")
print(f"Average Accuracy: {average_accuracy:.4f} (or {average_accuracy * 100:.2f}%)")
print("This is our measuring stick!")
