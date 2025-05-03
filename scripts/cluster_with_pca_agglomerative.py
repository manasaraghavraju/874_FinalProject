
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# === Load Call Type Matrix ===
df = pd.read_csv("data/processed/merged_spd_weather.csv", low_memory=False)
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()
df['Initial Call Type'] = df['Initial Call Type'].astype(str).str.lower().str.strip()
valid_df = df[~df['Neighborhood'].isin(['-', 'unknown']) & df['Neighborhood'].notna()]

call_type_matrix = (
    valid_df.groupby(['Neighborhood', 'Initial Call Type'])
    .size()
    .unstack(fill_value=0)
)

# === Normalize to Proportions ===
call_type_dist = call_type_matrix.div(call_type_matrix.sum(axis=1), axis=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(call_type_dist)

# === PCA ===
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# === Agglomerative Clustering ===
agglo = AgglomerativeClustering(n_clusters=4)
call_type_dist['pca_cluster'] = agglo.fit_predict(X_pca)
call_type_dist['total_calls'] = call_type_matrix.sum(axis=1)

# === Save Results ===
call_type_dist.reset_index().to_csv("output/neighborhood_pca_clusters.csv", index=False)
print("✅ Saved PCA-based clusters to output/neighborhood_pca_clusters.csv")
