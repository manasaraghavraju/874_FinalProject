
import pandas as pd

# === Load Data ===
df = pd.read_csv("data/processed/merged_spd_weather.csv", low_memory=False)
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()
df['Initial Call Type'] = df['Initial Call Type'].astype(str).str.lower().str.strip()

clusters = pd.read_csv("output/neighborhood_calltype_clusters.csv")
clusters["Neighborhood"] = clusters["Neighborhood"].str.lower().str.strip()

# === Merge to Associate Cluster Labels ===
df = df.merge(clusters[["Neighborhood", "call_type_cluster"]], on="Neighborhood", how="left")
df = df[df["call_type_cluster"].notna()]

# === Generate Summary ===
summary = []
for cluster in sorted(df["call_type_cluster"].unique()):
    cluster_df = df[df["call_type_cluster"] == cluster]
    neighborhood_count = cluster_df["Neighborhood"].nunique()
    total_calls = len(cluster_df)
    mean_calls = cluster_df.groupby("Neighborhood").size().mean()

    # Get top 3 call types
    top_calls = (
        cluster_df["Initial Call Type"]
        .value_counts()
        .head(3)
        .index
        .tolist()
    )
    summary.append({
        "Cluster": int(cluster),
        "Neighborhoods": neighborhood_count,
        "Avg Calls per Neighborhood": round(mean_calls, 2),
        "Top Call Types": ", ".join(top_calls)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("output/cluster_summary.csv", index=False)
print("✅ Summary saved to output/cluster_summary.csv")

# === Print Markdown Table ===
print("\n### Cluster Summary (Markdown Table)\n")
print(summary_df.to_markdown(index=False))
