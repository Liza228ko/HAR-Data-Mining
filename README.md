# Human Activity Recognition (HAR) — Data Mining Assignment 3

This repository contains the implementation, datasets preprocessing, model pipeline, and evaluation for the Human Activity Recognition (HAR) task on wrist-worn 3-axis accelerometer data.

## How to Run

### Directory Structure
Ensure your repository is organized as follows:
```text
/
├── data_processing.py      (Preprocesses raw accelerometer CSVs into structured features)
├── create_user_mapping.py  (Builds mapping of file IDs to user IDs for GroupKFold validation)
├── train_advanced.py       (Main script to train the final LightGBM model and generate predictions)
├── train_baseline.py       (Trains a simple Random Forest baseline on time-domain stats under random splits)
├── baseline_simple.py      (Compares baseline classifier configurations under GroupKFold)
├── ablation_study.py       (Runs the ablation experiments of feature groups under GroupKFold)
├── report.md               (This report document)
├── DM_asg3_111550205.pdf   (Compiled report document)
├── class_distribution.png  (Visual chart of training label distribution)
├── architecture.png        (Visual diagram of sequence windowing and model pipeline)
└── data/
    ├── user_mapping_train.csv
    ├── train/
    │   └── train/
    │       └── User_xxx/ (Raw CSVs)
    └── test/
        └── test/
            └── User_xxx/ (Raw CSVs)
```

### Dependencies
Install the required packages via pip:
```bash
pip install pandas numpy scikit-learn lightgbm tqdm joblib scipy
```

### Execution Steps
1. **Extract Features**: Run the feature extraction script to preprocess the raw CSV logs:
   ```bash
   python3 data_processing.py
   ```
   This reads the raw files and creates `data/train_features_advanced.csv` and `data/test_features_advanced.csv`.

2. **Generate User Mapping**: Build the user-to-file mapping for GroupKFold validation:
   ```bash
   python3 create_user_mapping.py
   ```
   This generates `data/user_mapping_train.csv`.

3. **Run Baseline & Ablation Evaluations (Optional)**:
   * To reproduce the baseline comparison runs:
     ```bash
     python3 baseline_simple.py
     ```
   * To train and evaluate the original simple Random Forest baseline (Macro F1 = 0.7002 on Stratified KFold):
     ```bash
     python3 train_baseline.py
     ```
   * To reproduce the exact ablation study scores:
     ```bash
     python3 ablation_study.py
     ```

4. **Train Advanced Model & Predict**: Run the main advanced training script to evaluate the CV score and output predictions:
   ```bash
   python3 train_advanced.py
   ```
   This trains the final model on all data and saves the predictions to `data/submission_advanced.csv`.
