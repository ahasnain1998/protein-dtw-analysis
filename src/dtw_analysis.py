import os
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from tqdm import tqdm

def select_contact_columns(file_path):
    df = pd.read_csv(file_path)
    contact_columns = [col for col in df.columns if col.startswith('# Contacts') or not col.startswith('Frame.')]
    return df[contact_columns]

def calculate_dtw_distance(data1, data2):
    data1_tuples = [tuple(row) for row in data1.to_numpy()]
    data2_tuples = [tuple(row) for row in data2.to_numpy()]
    distance, path = fastdtw(data1_tuples, data2_tuples, dist=euclidean)
    return distance / len(path)

def compute_distance_matrix(folder_path):
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    data_list = [select_contact_columns(os.path.join(folder_path, f)) for f in tqdm(csv_files, desc="Loading Contact Maps")]
    dist_matrix = pd.DataFrame(index=csv_files, columns=csv_files, dtype=float)

    for i in tqdm(range(len(data_list)), desc="Computing DTW Distances"):
        for j in range(i, len(data_list)):
            dist = calculate_dtw_distance(data_list[i], data_list[j])
            dist_matrix.iloc[i, j] = dist_matrix.iloc[j, i] = dist

    dist_matrix.fillna(0, inplace=True)
    return dist_matrix