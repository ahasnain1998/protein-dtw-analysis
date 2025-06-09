import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

 

from src.preproccessing import merge_and_binarize
from src.dtw_analysis import compute_distance_matrix
from src.clustering import run_kmeans_clustering, run_hierarchical_clustering
import pandas as pd
import os
from time import time

base_data_dir = 'data/11-WT'
base_data_dir1 = 'data/labels'
# Define paths
folder_A = os.path.join(base_data_dir, 'contacts_chain_A')
folder_B = os.path.join(base_data_dir, 'contacts_chain_B')
output_folder = os.path.join('output', 'merged_binarized')
labels_path = os.path.join(base_data_dir1, 'labels.csv')
distance_output_path = os.path.join('output', 'distance_matrix.csv')
os.makedirs('output', exist_ok=True)

# === Step 1: Merge and Binarize ===
t1 = time()
merge_and_binarize(folder_A, folder_B, output_folder)
print(f"⏱️ Merge + Binarize Time: {time() - t1:.2f} seconds")

# === Step 2: Compute DTW Distance Matrix ===
t2 = time()
distance_matrix = compute_distance_matrix(output_folder)
distance_matrix.to_csv(distance_output_path)
print(f"⏱️ DTW Matrix Time: {time() - t2:.2f} seconds")
# === Step 3: Load labels ===
labels_df = pd.read_csv(labels_path, header=None, names=["Trajectory", "Type", "Category"])
labels_df = labels_df.set_index("Trajectory")
labels_df = labels_df.loc[list(distance_matrix.index)]  # align order
labels = labels_df["Type"].tolist()

# === Step 4: Run KMeans Clustering ===
t3 = time()
mds_coords, kmeans_labels, kmeans_silhouette, kmeans_dbi = run_kmeans_clustering(
    distance_matrix.values, n_clusters=len(set(labels)), labels=labels
)
print(f"KMeans - Silhouette Score: {kmeans_silhouette:.4f}, Davies-Bouldin Index: {kmeans_dbi:.4f}")

# === Step 5: Run Hierarchical Clustering ===
linkage_matrix, hierarchical_labels, hier_silhouette, hier_dbi = run_hierarchical_clustering(
    distance_matrix.values, labels_df, n_clusters=len(set(labels))
)
print(f"Hierarchical - Silhouette Score: {hier_silhouette:.4f}, Davies-Bouldin Index: {hier_dbi:.4f}")
print(f"Clustering + Plotting Time: {time() - t3:.2f} seconds")

print("Clustering and validation complete. Plots saved to output/.")

