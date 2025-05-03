import pandas as pd

# === CONFIG ===
MERGED_DATA_PATH = "data/processed/merged_spd_weather.csv"
CLUSTER_CSV_PATH = "output/neighborhood_hdbscan_clusters.csv"
OUTPUT_CSV = "output/hdbscan_outlier_summary.csv"

# === LOAD DATA ===
df = pd.read_csv(MERGED_DATA_PATH, low_memory=False)
clusters = pd.read_csv(CLUSTER_CSV_PATH)

# === CLEAN + NORMALIZE NEIGHBORHOOD NAMES ===
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()
df = df[~df['Neighborhood'].isin(['-', 'unknown']) & df['Neighborhood'].notna()]
clusters['Neighborhood'] = clusters['Neighborhood'].astype(str).str.lower().str.strip()

# === FILTER OUTLIERS (HDBSCAN CLUSTER -1) ===
outlier_neighborhoods = clusters[clusters['hdbscan_cluster'] == -1]['Neighborhood'].unique()
outlier_df = df[df['Neighborhood'].isin(outlier_neighborhoods)]

# === GET TOP 3 CALL TYPES PER NEIGHBORHOOD ===
top_calls = (
    outlier_df.groupby(['Neighborhood', 'Initial Call Type'])
    .size()
    .reset_index(name='count')
    .sort_values(['Neighborhood', 'count'], ascending=[True, False])
)

top_calls_summary = (
    top_calls.groupby('Neighborhood')
    .head(3)
    .groupby('Neighborhood')['Initial Call Type']
    .apply(lambda x: ', '.join(x))
    .reset_index()
    .rename(columns={'Initial Call Type': 'Top Call Types'})
)

# === TOTAL CALL COUNT PER OUTLIER NEIGHBORHOOD ===
call_counts = outlier_df['Neighborhood'].value_counts().reset_index()
call_counts.columns = ['Neighborhood', 'Total Calls']

# === MERGE RESULTS ===
summary_df = pd.merge(call_counts, top_calls_summary, on='Neighborhood', how='left')

# === SAVE OUTPUT ===
summary_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Outlier summary saved to {OUTPUT_CSV}")
