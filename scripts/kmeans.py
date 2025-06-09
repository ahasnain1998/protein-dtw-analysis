import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# --- File paths ---
distance_matrix_file = "output/distance_matrix.csv"
labels_file = "data/labels/labels.csv"

# --- Load data ---
distance_matrix = pd.read_csv(distance_matrix_file, index_col=0)
trajectory_names = distance_matrix.index.tolist()

# Load and align labels
labels = pd.read_csv(labels_file, header=None, names=["Trajectory", "Type", "Category"])
labels = labels.set_index("Trajectory")
labels = labels.loc[trajectory_names]  # ensure correct order

# Sanity check
if len(labels) != len(trajectory_names):
    raise ValueError("Mismatch between trajectory names and label rows")

# --- Preprocess distance matrix ---
distance_matrix = distance_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
np.fill_diagonal(distance_matrix.values, 0)
distance_matrix = (distance_matrix + distance_matrix.T) / 2

# --- Normalize ---
scaler = StandardScaler()
normalized_distance_matrix = scaler.fit_transform(distance_matrix.values)
normalized_distance_matrix = (normalized_distance_matrix + normalized_distance_matrix.T) / 2

# --- MDS for projection ---
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_transformed = mds.fit_transform(normalized_distance_matrix)

# --- KMeans clustering ---
n_clusters = len(set(labels["Type"]))  # Detect number of label groups
kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
kmeans_labels = kmeans.fit_predict(mds_transformed)

# --- Evaluate with ARI ---
true_labels = labels["Type"].astype("category").cat.codes
ari = adjusted_rand_score(true_labels, kmeans_labels)
print(f"Adjusted Rand Index (KMeans vs. Ground Truth): {ari:.4f}")

# --- Create DataFrame for plotting ---
plot_df = pd.DataFrame({
    "MDS1": mds_transformed[:, 0],
    "MDS2": mds_transformed[:, 1],
    "Cluster": kmeans_labels,
    "Trajectory": trajectory_names,
    "Type": labels["Type"].values,
    "Category": labels["Category"].values
})

# Sort by label type
plot_df["Type"] = pd.Categorical(plot_df["Type"], categories=sorted(set(labels["Type"])), ordered=True)
plot_df = plot_df.sort_values("Type")

# --- Plot ---
y_max = max(abs(plot_df["MDS2"].min()), abs(plot_df["MDS2"].max())) + 0.1
fig = px.scatter(
    plot_df,
    x="MDS1", y="MDS2", color="Type",
    hover_data={"Trajectory": True, "Type": True, "Category": True},
    title=f"KMeans Clustering (n={n_clusters}) with Hover Info",
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig.update_layout(
    legend=dict(
        title="Label Types",
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    ),
    autosize=False,
    width=800,
    height=800,
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, range=[-y_max, y_max]),
    margin=dict(l=50, r=50, b=50, t=50, pad=10),
    shapes=[
        dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="black", width=2)
        )
    ]
)

# --- Show plot ---
fig.show()
fig.write_image("output/kmeans_plot.png")