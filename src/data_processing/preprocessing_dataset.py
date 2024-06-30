import pandas as pd
import numpy as np
import os
import json
import csv
from pathlib import Path

# Paths
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = Path(SRC_PATH).parent.parent
DATASETS_PATH = os.path.join(ROOT_PATH, 'dataset/fault')
OUTPUTS_PATH = os.path.join(ROOT_PATH, 'outputs')
PLOTS_PATH = os.path.join(ROOT_PATH, 'plots')

#initial_dataset = 'iase/can_messages.log'
initial_dataset = 'feature_time_48k_2048_load_1.csv'
file_path = os.path.join(DATASETS_PATH, initial_dataset)


def map_faults(df):
    # Create a dictionary to map the faults
    faults = df['fault'].unique()
    faults = np.sort(faults)

    fault_dict = {faults[i]: i for i in range(len(faults))}
    print(fault_dict)

    # Map the faults
    df['fault'] = df['fault'].map(fault_dict)

    return df


df = pd.read_csv(file_path)
print(df['fault'].value_counts())
df = map_faults(df)

df = df.drop(columns=['rms'])
    
df.to_csv(os.path.join(DATASETS_PATH, 'mapped_dataset_2.csv'), index=False)

