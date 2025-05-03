
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
from collections import Counter
import os

# Load merged call data
df = pd.read_csv("data/processed/merged_spd_weather.csv", low_memory=False)
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()
df = df[~df['Neighborhood'].isin(['-', 'unknown']) & df['Neighborhood'].notna()]

# Aggregate call types per neighborhood
call_type_counts = (
    df.groupby(['Neighborhood', 'Initial Call Type'])
      .size()
      .unstack(fill_value=0)
)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(call_type_counts)

# PCA for dimensionality reduction (2D for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, prediction_data=True)
labels = clusterer.fit_predict(X_pca)

# Prepare results
call_type_counts["hdbscan_cluster"] = labels
call_type_counts["total_calls"] = call_type_counts.drop(columns=["hdbscan_cluster"]).sum(axis=1)
call_type_counts.reset_index(inplace=True)

# Save results
os.makedirs("output", exist_ok=True)
call_type_counts.to_csv("output/neighborhood_hdbscan_clusters.csv", index=False)

# Generate cluster summary
summary = call_type_counts.groupby("hdbscan_cluster").agg(
    Neighborhoods=("Neighborhood", "count"),
    AvgCalls=("total_calls", "mean")
).reset_index()

# Top call types per cluster
top_calls = (
    df.groupby(['Neighborhood', 'Initial Call Type'])
      .size()
      .reset_index(name='count')
)

top_clusters = call_type_counts[['Neighborhood', 'hdbscan_cluster']]
top_merged = top_calls.merge(top_clusters, on='Neighborhood', how='left')

top_call_summary = (
    top_merged.groupby(['hdbscan_cluster', 'Initial Call Type'])['count']
    .sum()
    .reset_index()
    .sort_values(['hdbscan_cluster', 'count'], ascending=[True, False])
    .groupby('hdbscan_cluster')
    .head(3)
)

call_type_map = (
    top_call_summary.groupby("hdbscan_cluster")["Initial Call Type"]
    .apply(lambda x: ", ".join(x))
    .to_dict()
)

summary["Top Call Types"] = summary["hdbscan_cluster"].map(call_type_map)

# Save summary
summary.to_csv("output/hdbscan_cluster_summary.csv", index=False)
print("✅ HDBSCAN clustering complete. Results saved to output/neighborhood_hdbscan_clusters.csv and summary to hdbscan_cluster_summary.csv.")
