# Directory: scripts/visualize_hierarchical_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.patches as mpatches
import os

# --- Load distance matrix and labels ---
distance_matrix_file = "output/distance_matrix.csv"
labels_file = "data/labels/labels.csv"

# Load distance matrix
distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
trajectory_names = distance_matrix.index.tolist()

# Load and align labels
labels = pd.read_csv(labels_file, header=None, names=["Trajectory", "Type", "Category"])
labels = labels.set_index("Trajectory")
labels = labels.loc[trajectory_names]  # align order

# Sanity check
if len(labels) != len(trajectory_names):
    raise ValueError("Mismatch between trajectory names and label rows")

# Preprocess distance matrix
distance_matrix = distance_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
np.fill_diagonal(distance_matrix.values, 0)
distance_matrix = (distance_matrix + distance_matrix.T) / 2

# Normalize
scaler = StandardScaler()
normalized = scaler.fit_transform(distance_matrix)
normalized = (normalized + normalized.T) / 2

# --- Agglomerative Clustering ---
n_clusters = len(set(labels['Type']))
model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
cluster_labels = model.fit_predict(distance_matrix.values)

# Evaluate
true_labels = labels["Type"].astype("category").cat.codes
ari = adjusted_rand_score(true_labels, cluster_labels)
print(f"Adjusted Rand Index (Hierarchical vs. Ground Truth): {ari:.4f}")

# --- Dendrogram with colored leaf labels and legend ---
linkage_matrix = linkage(squareform(distance_matrix), method='complete')
os.makedirs("output", exist_ok=True)

# Color mapping
unique_types = sorted(set(labels["Type"]))
label_to_color = {label: plt.cm.tab10(i) for i, label in enumerate(unique_types)}
leaf_colors = [label_to_color[labels.loc[traj]['Type']] for traj in trajectory_names]

plt.figure(figsize=(14, 8))
dendro = dendrogram(
    linkage_matrix,
    labels=trajectory_names,
    leaf_rotation=90,
    leaf_font_size=10,
    link_color_func=lambda k: 'black'
)

ax = plt.gca()
x_labels = ax.get_xmajorticklabels()
for lbl in x_labels:
    traj = lbl.get_text()
    lbl.set_color(label_to_color[labels.loc[traj]['Type']])

legend_patches = [mpatches.Patch(color=color, label=label) for label, color in label_to_color.items()]
plt.legend(handles=legend_patches, title="True Labels", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")
plt.xlabel("Trajectories")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("output/hierarchical_dendrogram.png")
plt.show()

# Save cluster labels
labels_out = pd.DataFrame({
    "Trajectory": trajectory_names,
    "TrueType": labels["Type"].values,
    "ClusterLabel": cluster_labels
})
labels_out.to_csv("output/hierarchical_cluster_labels.csv", index=False)