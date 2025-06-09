import os
import pandas as pd
import re
from tqdm import tqdm

def extract_number(file_name):
    match = re.search(r'(\d+)', file_name)
    return match.group(1) if match else None

def binarize_dataframe(df):
    residue_cols = [col for col in df.columns if not col.startswith("Frame.")]
    df[residue_cols] = (df[residue_cols] != 0).astype(int)
    return df

def merge_and_binarize(folder_A, folder_B, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files_A = {extract_number(f): f for f in os.listdir(folder_A) if f.endswith('.csv')}
    files_B = {extract_number(f): f for f in os.listdir(folder_B) if f.endswith('.csv')}

    for number in tqdm(sorted(set(files_A) & set(files_B)), desc="Merging and Binarizing"):
        path_A = os.path.join(folder_A, files_A[number])
        path_B = os.path.join(folder_B, files_B[number])

        df_A = pd.read_csv(path_A)
        df_B = pd.read_csv(path_B)
        # ✅ Fixed: Add suffixes to avoid duplicate column names
        df = pd.concat([df_A.add_suffix('_A'), df_B.add_suffix('_B')], axis=1)

        df = binarize_dataframe(df)
        output_path = os.path.join(output_folder, f'contacts_chain_AB_{number}.csv')
        df.to_csv(output_path, index=False)

    print(f"✅ Merged and binarized files saved to: {output_folder}")
