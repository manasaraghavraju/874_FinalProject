
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import os

# === CONFIG ===
MERGED_DATA_PATH = "data/processed/merged_spd_weather.csv"
OUTPUT_DIR = "output"
CLUSTER_CSV_PATH = os.path.join(OUTPUT_DIR, "neighborhood_gmm_clusters.csv")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "gmm_cluster_summary.csv")

# === LOAD AND CLEAN DATA ===
df = pd.read_csv(MERGED_DATA_PATH, low_memory=False)
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()
df = df[~df['Neighborhood'].isin(['-', 'unknown']) & df['Neighborhood'].notna()]

# === BUILD CALL TYPE FREQUENCY MATRIX ===
call_matrix = pd.crosstab(df['Neighborhood'], df['Initial Call Type'])
call_matrix = call_matrix.loc[:, (call_matrix != 0).any(axis=0)]  # drop all-zero columns

# === NORMALIZE AND REDUCE DIMENSIONS ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(call_matrix)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# === GMM CLUSTERING ===
gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(X_pca)

# === SAVE CLUSTER ASSIGNMENTS ===
cluster_df = pd.DataFrame({
    "Neighborhood": call_matrix.index,
    "gmm_cluster": clusters
})
cluster_df.to_csv(CLUSTER_CSV_PATH, index=False)
print(f"✅ GMM cluster labels saved to {CLUSTER_CSV_PATH}")

# === GENERATE CLUSTER SUMMARY ===
merged = pd.merge(df, cluster_df, left_on='Neighborhood', right_on='Neighborhood')

top_calls = (
    merged.groupby(['gmm_cluster', 'Initial Call Type'])
    .size()
    .reset_index(name='count')
    .sort_values(['gmm_cluster', 'count'], ascending=[True, False])
)

top_calls_ranked = (
    top_calls.groupby('gmm_cluster')
    .head(3)
    .groupby('gmm_cluster')['Initial Call Type']
    .apply(lambda x: ', '.join(x))
    .reset_index()
    .rename(columns={'Initial Call Type': 'Top Call Types'})
)

call_counts = merged.groupby('gmm_cluster')['Neighborhood'].nunique().reset_index(name='Neighborhoods')
avg_volume = merged.groupby('gmm_cluster')['Neighborhood'].value_counts().groupby('gmm_cluster').mean().reset_index(name='Avg Calls per Neighborhood')

summary = pd.merge(call_counts, avg_volume, on='gmm_cluster')
summary = pd.merge(summary, top_calls_ranked, on='gmm_cluster')
summary.to_csv(SUMMARY_CSV_PATH, index=False)

print(f"📊 GMM cluster summary saved to {SUMMARY_CSV_PATH}")
