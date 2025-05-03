
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
import branca.colormap as cm
import os

# === CONFIG ===
GEOJSON_PATH = "data/raw/spd_dispatch_neighborhoods.geojson"
CLUSTERS_CSV = "output/neighborhood_gmm_bic_clusters.csv"
SUMMARY_CSV = "output/gmm_bic_cluster_summary.csv"
OUTPUT_MAP = "output/interactive_gmm_cluster_map.html"

# === LOAD DATA ===
gdf = gpd.read_file(GEOJSON_PATH)
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf['Neighborhood'] = gdf['Neighborhood'].str.lower().str.strip()
gdf = gdf.to_crs(epsg=4326)

clusters_df = pd.read_csv(CLUSTERS_CSV)
clusters_df['Neighborhood'] = clusters_df['Neighborhood'].str.lower().str.strip()

# === MERGE CLUSTER LABELS ===
gdf = gdf.merge(clusters_df, on="Neighborhood", how="left")
gdf['gmm_cluster'] = gdf['gmm_cluster'].fillna(-1).astype(int)

# === COLORMAP ===
n_clusters = gdf['gmm_cluster'].nunique()
palette = cm.linear.Set1_09.scale(0, n_clusters - 1)
palette.caption = "GMM Cluster Assignment"

# === MAP SETUP ===
m = folium.Map(location=[47.6, -122.33], zoom_start=12, tiles="CartoDB positron")

# === ADD NEIGHBORHOODS ===
def style_func(feature):
    val = feature['properties']['gmm_cluster']
    if val == -1:
        return {"fillOpacity": 0.1, "color": "black", "weight": 1}
    return {
        "fillColor": palette(val),
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.7
    }

folium.GeoJson(
    gdf,
    tooltip=GeoJsonTooltip(fields=["Neighborhood", "gmm_cluster"], aliases=["Neighborhood:", "GMM Cluster:"]),
    style_function=style_func
).add_to(m)

palette.add_to(m)
folium.LayerControl().add_to(m)

# === SAVE MAP ===
m.save(OUTPUT_MAP)
print(f"✅ GMM cluster map saved to {OUTPUT_MAP}")
