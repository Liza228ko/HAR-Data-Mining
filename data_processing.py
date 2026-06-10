import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.stats as stats

def extract_features_from_df(df, prefix=''):
    features = {}
    sensor_cols = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']
    
    for col in sensor_cols:
        # Time domain
        features[f'{prefix}{col}_mean'] = df[col].mean()
        features[f'{prefix}{col}_std'] = df[col].std()
        features[f'{prefix}{col}_max'] = df[col].max()
        features[f'{prefix}{col}_min'] = df[col].min()
        features[f'{prefix}{col}_median'] = df[col].median()
        features[f'{prefix}{col}_skew'] = df[col].skew()
        features[f'{prefix}{col}_kurtosis'] = df[col].kurtosis()
        
        # Frequency domain
        fft_vals = np.abs(np.fft.fft(df[col]))
        fft_vals = fft_vals[1:len(fft_vals)//2] # skip DC and take first half
        if len(fft_vals) > 0:
            features[f'{prefix}{col}_fft_mean'] = np.mean(fft_vals)
            features[f'{prefix}{col}_fft_std'] = np.std(fft_vals)
            features[f'{prefix}{col}_fft_max'] = np.max(fft_vals)
            features[f'{prefix}{col}_fft_energy'] = np.sum(fft_vals ** 2) / len(fft_vals)
            features[f'{prefix}{col}_fft_dominant'] = np.argmax(fft_vals)
        else:
            features[f'{prefix}{col}_fft_mean'] = 0.0
            features[f'{prefix}{col}_fft_std'] = 0.0
            features[f'{prefix}{col}_fft_max'] = 0.0
            features[f'{prefix}{col}_fft_energy'] = 0.0
            features[f'{prefix}{col}_fft_dominant'] = 0.0

    # Jerk Signals (Rate of change of acceleration)
    for col in ['mean_x', 'mean_y', 'mean_z']:
        jerk = df[col].diff().fillna(0)
        features[f'{prefix}{col}_jerk_mean'] = jerk.mean()
        features[f'{prefix}{col}_jerk_std'] = jerk.std()
        features[f'{prefix}{col}_jerk_max'] = jerk.max()
        features[f'{prefix}{col}_jerk_min'] = jerk.min()

    # Inter-axis correlation
    features[f'{prefix}corr_mean_xy'] = df['mean_x'].corr(df['mean_y'])
    features[f'{prefix}corr_mean_xz'] = df['mean_x'].corr(df['mean_z'])
    features[f'{prefix}corr_mean_yz'] = df['mean_y'].corr(df['mean_z'])
    
    # Signal Magnitude Area (SMA)
    sma = np.sqrt(df['mean_x']**2 + df['mean_y']**2 + df['mean_z']**2)
    features[f'{prefix}sma_mean'] = sma.mean()
    features[f'{prefix}sma_std'] = sma.std()
    features[f'{prefix}sma_max'] = sma.max()
    features[f'{prefix}sma_min'] = sma.min()
    
    return features

def extract_features(file_path, is_test=False):
    """
    Extracts advanced features using overall stats + 3 sub-windows.
    Matches the logic that achieved Kaggle 0.7779.
    """
    try:
        df = pd.read_csv(file_path)
        features = {}
        
        # 1. Overall Features
        features.update(extract_features_from_df(df, prefix='overall_'))
        
        # 2. Sub-windowing Features (3 equal chunks of 100s each)
        n_rows = len(df)
        if n_rows > 0:
            chunk_size = max(1, n_rows // 3)
            chunks = [
                df.iloc[0:chunk_size], 
                df.iloc[chunk_size:2*chunk_size], 
                df.iloc[2*chunk_size:]
            ]
            for i, chunk in enumerate(chunks):
                if len(chunk) > 0:
                    features.update(extract_features_from_df(chunk, prefix=f'w{i+1}_'))
                
        # Handle any NaNs
        for k in features:
            if pd.isna(features[k]):
                features[k] = 0.0
            
        if not is_test:
            features['label'] = int(df['label'].iloc[0])
            
        features['file_id'] = int(df['file_id'].iloc[0])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    train_files = glob.glob('data/train/train/*/*.csv')
    test_files = glob.glob('data/test/test/*/*.csv')
    
    print(f"Starting feature extraction for {len(train_files)} training files...")
    train_features = Parallel(n_jobs=-1)(delayed(extract_features)(f, False) for f in tqdm(train_files, desc="Train Data"))
    train_features = [f for f in train_features if f is not None]
    
    train_df = pd.DataFrame(train_features)
    train_df.to_csv('data/train_features_advanced.csv', index=False)
    print(f"Saved {len(train_df)} rows with {len(train_df.columns)} features to data/train_features_advanced.csv")
    
    print(f"\nStarting feature extraction for {len(test_files)} testing files...")
    test_features = Parallel(n_jobs=-1)(delayed(extract_features)(f, True) for f in tqdm(test_files, desc="Test Data"))
    test_features = [f for f in test_features if f is not None]
    
    test_df = pd.DataFrame(test_features)
    test_df.to_csv('data/test_features_advanced.csv', index=False)
    print(f"Saved {len(test_df)} rows with {len(test_df.columns)} features to data/test_features_advanced.csv")
    print("\nFeature Extraction Complete!")
