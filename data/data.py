import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# Function for preprocessing the dataset
def data_preprocess(data):

    # Drop 'Date' and 'Time' columns
    data = data.drop(['Date', 'Time'], axis=1)

    # Normalize features
    scaler = MinMaxScaler()
    features = data.drop(['Room_Occupancy_Count'], axis=1)
    features_normalized = scaler.fit_transform(features)

    # Combine normalized features with target
    data_normalized = pd.DataFrame(features_normalized, columns=features.columns)
    data_normalized['Room_Occupancy_Count'] = data['Room_Occupancy_Count'].values

    return data_normalized

# Function for creating sequences from the data
def create_sequences(data, sequence_length):
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length, :-1].values)
        y.append(data.iloc[i + sequence_length, -1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

# Datse class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]