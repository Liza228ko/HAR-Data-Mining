# Data Mining Assignment 3: Human Activity Recognition (HAR)
**Student ID:** 111550205  
**Name:** Yelyzaveta Kozachenko

---

## 1. Preliminary Analysis
The dataset consists of 3-axis accelerometer readings from 100 users, aggregated into 1-second intervals. Each sample is a 5-minute sequence (300 seconds).

### Raw Data Observations
- **Dominant Class:** Activity 0 (likely "Standing" or "Still") accounts for approximately 42% of the training data. This class imbalance is a major challenge, as naive models might overfit to the majority class.
- **Signal Characteristics:** Activities like walking (likely labels 1, 3, 5) show periodic oscillations in the `mean_x/y/z` channels, while stationary activities have very low `std_x/y/z` values.
- **Gravity Component:** The measurements include gravity ($1g \approx 9.81 m/s^2$). The orientation of the wrist significantly affects the baseline mean values of each axis.

### Naive Baseline Performance
A Random Forest model trained on simple time-domain features (Mean, Std, Max, Min, Median) achieved a **Macro F1-score of 0.7002**. While the accuracy was high (~88%), the low Macro F1 indicated that the model struggled with minority classes like Label 2 (F1 ~0.30).

---

## 2. Preprocessing & Feature Engineering
To improve performance, I implemented a robust feature extraction pipeline that transforms the 300-row sequences into a single feature vector.

### Techniques Used:
1.  **Time-Domain Stats:** Beyond Mean/Std, I added **Skewness** and **Kurtosis** to capture the distribution shape of the acceleration.
2.  **Frequency-Domain (FFT):** I applied Fast Fourier Transform to each axis. I extracted the **FFT Mean, Std, Max, Energy, and Dominant Frequency**. This is critical for distinguishing rhythmic activities (walking) from random movements.
3.  **Jerk Signals:** Calculated the derivative of acceleration ($da/dt$) for each axis. This captures the "suddenness" of movements.
4.  **Inter-axis Correlation:** Calculated the correlation between X, Y, and Z axes to capture the 3D movement patterns.

### Improvement Summary:
| Technique | Macro F1 (Local CV) | Improvement |
| :--- | :--- | :--- |
| Baseline (Time Only) | 0.7002 | - |
| Advanced (Time + Freq + Jerk) | 0.7621 | +0.0619 |
| Final Model (LightGBM + Class Weighting) | **0.7779 (Kaggle)** | **+0.0777** |

---

## 3. Sequential Alignment & Temporal Dependencies
![System Architecture](architecture.svg)

Each activity label is assigned to a 5-minute window. To capture the temporal nature of the data:
- **Statistical Aggregation:** Instead of treating the 300 seconds as independent, I aggregated them into global window statistics.
- **Sequence Context:** The Jerk signal explicitly captures the relationship between consecutive seconds.
- **Dominant Frequency:** The FFT dominant frequency captures the "tempo" of the activity across the entire 5-minute window, effectively "aligning" the periodic signals with the label.

---

## 4. Ablation Study
I performed an ablation study to evaluate the impact of the core design choices:

| Experiment | Configuration | Macro F1 | Impact |
| :--- | :--- | :--- | :--- |
| **A** | Baseline (Random Forest, Time-Domain) | 0.7002 | Base |
| **B** | LightGBM + Frequency Features | 0.7415 | +0.0413 |
| **C** | LightGBM + Freq + Jerk + Correlation | 0.7550 | +0.0135 |
| **D** | **Full Model (C + Balanced Class Weights)** | **0.7621** | **+0.0071** |

**Key Findings:** 
- The **Frequency Features** provided the largest jump in performance, as they help the model see the "vibration" patterns in the wrist.
- **Class Weighting** was essential for boosting the F1-score of rare activities (Labels 2 and 4), even if it slightly lowered overall accuracy.

---

## 5. How to Run
### Directory Structure
To ensure the code runs successfully, please organize the files as follows:
```text
/
├── data_processing.py
├── train_advanced.py
├── train_baseline.py
├── report.md
├── architecture.svg
└── data/
    ├── user_mapping_train.csv (Provided in Repo)
    ├── train/
    │   └── train/
    │       └── User_xxx/ (Raw CSVs)
    └── test/
        └── test/
            └── User_xxx/ (Raw CSVs)
```

### Dependencies:
```bash
pip install pandas numpy scikit-learn lightgbm tqdm joblib
```

### Execution Steps:
1.  **Feature Extraction:** Run `python3 data_processing.py`. This will read the raw CSVs from `data/train/` and generate `data/train_features_advanced.csv`.
2.  **Training & Submission:** Run `python3 train_advanced.py`. This trains the LightGBM model and generates `data/submission_advanced.csv`.

**GitHub Repository:** [https://github.com/Liza228ko/HAR-Data-Mining](https://github.com/Liza228ko/HAR-Data-Mining)

---

## 6. Conclusion
By engineering a combination of time and frequency domain features, I was able to capture both the intensity and the rhythm of human activities. The use of LightGBM with balanced class weights significantly improved the Macro F1-score on minority activities, leading to a Public Leaderboard score of **0.7779**, which successfully surpassed all three competition baselines.
