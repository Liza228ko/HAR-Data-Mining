import glob
import pandas as pd
import os
from tqdm import tqdm

def create_mapping(path_pattern):
    files = glob.glob(path_pattern)
    mapping = []
    for f in tqdm(files):
        # Path format: data/train/train/User_001/1.csv
        parts = f.split('/')
        user_id = int(parts[-2].split('_')[1])
        # Read file_id from inside the CSV (more reliable than filename)
        df = pd.read_csv(f, usecols=['file_id'], nrows=1)
        file_id = int(df['file_id'].iloc[0])
        mapping.append({'file_id': file_id, 'user_id': user_id})
    return mapping

print("Creating User Mapping for Train Data...")
train_map = create_mapping('data/train/train/*/*.csv')
df_train = pd.DataFrame(train_map)
df_train.to_csv('data/user_mapping_train.csv', index=False)

print("Creating User Mapping for Test Data...")
test_map = create_mapping('data/test/test/*/*.csv')
df_test = pd.DataFrame(test_map)
df_test.to_csv('data/user_mapping_test.csv', index=False)

print("Done! Mapping saved to data/user_mapping_train.csv and data/user_mapping_test.csv")
