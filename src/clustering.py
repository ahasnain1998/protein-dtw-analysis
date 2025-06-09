# Directory: src/clustering.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm

def run_kmeans_clustering(distance_matrix, n_clusters=2, labels=None, output_dir="output"):
    scaler = StandardScaler()
    normalized = scaler.fit_transform(distance_matrix)
    normalized = (normalized + normalized.T) / 2

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_transformed = mds.fit_transform(normalized)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(mds_transformed)

    silhouette = silhouette_score(normalized, cluster_labels)
    dbi = davies_bouldin_score(mds_transformed, cluster_labels)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for label in np.unique(cluster_labels):
        idx = cluster_labels == label
        plt.scatter(mds_transformed[idx, 0], mds_transformed[idx, 1], label=f"Cluster {label}")
    plt.title("KMeans Clustering (MDS Projection)")
    plt.xlabel("MDS1")
    plt.ylabel("MDS2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kmeans_mds_plot.png"))
    plt.close()

    if labels is not None:
        true_labels = pd.Series(labels).astype("category").cat.codes
        ari = adjusted_rand_score(true_labels, cluster_labels)
        print(f"Adjusted Rand Index (KMeans vs. Ground Truth): {ari:.4f}")

    return mds_transformed, cluster_labels, silhouette, dbi

def run_hierarchical_clustering(distance_matrix, labels_df, n_clusters=2, method='complete', output_dir="output"):
    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0)
    os.makedirs(output_dir, exist_ok=True)

    # Agglomerative clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=method)
    cluster_labels = model.fit_predict(distance_matrix)

    # Normalize for MDS
    scaler = StandardScaler()
    normalized = scaler.fit_transform(distance_matrix)
    normalized = (normalized + normalized.T) / 2
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_transformed = mds.fit_transform(normalized)

    # Evaluation
    true_labels = labels_df["Type"].astype("category").cat.codes
    silhouette = silhouette_score(normalized, cluster_labels)
    dbi = davies_bouldin_score(mds_transformed, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index (Hierarchical vs. Ground Truth): {ari:.4f}")

    # Dendrogram coloring based on labels (not trajectories)
    linkage_matrix = linkage(squareform(distance_matrix), method=method)
    trajectory_names = labels_df.index.tolist()
    label_to_color = {label: plt.cm.tab10(i) for i, label in enumerate(sorted(set(labels_df['Type'])))}
    label_colors = [label_to_color[labels_df.loc[traj]['Type']] for traj in trajectory_names]

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
        lbl.set_color(label_to_color[labels_df.loc[traj]['Type']])

    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in label_to_color.items()]
    plt.legend(handles=legend_patches, title="True Labels", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.title(f"Hierarchical Clustering Dendrogram ({method})")
    plt.xlabel("Trajectories")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hierarchical_dendrogram.png"))
    plt.close()

    # Save cluster labels
    labels_out = pd.DataFrame({
        "Trajectory": trajectory_names,
        "TrueType": labels_df["Type"].values,
        "ClusterLabel": cluster_labels
    })
    labels_out.to_csv(os.path.join(output_dir, "hierarchical_cluster_labels.csv"), index=False)

    return linkage_matrix, cluster_labels, silhouette, dbi