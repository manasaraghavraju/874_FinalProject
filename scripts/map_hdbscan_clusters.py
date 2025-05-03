
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
import branca.colormap as cm
import os

# === Load Data ===
clusters = pd.read_csv("output/neighborhood_hdbscan_clusters.csv")
gdf = gpd.read_file("data/raw/spd_dispatch_neighborhoods.geojson")
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf['Neighborhood'] = gdf['Neighborhood'].str.lower().str.strip()
clusters['Neighborhood'] = clusters['Neighborhood'].str.lower().str.strip()

# Merge spatial and cluster data
gdf = gdf.merge(clusters[['Neighborhood', 'hdbscan_cluster']], on="Neighborhood", how="left")
gdf = gdf.to_crs(epsg=4326)

# === Create Map ===
m = folium.Map(location=[47.6, -122.33], zoom_start=12, tiles='CartoDB positron')
cluster_layer = folium.FeatureGroup(name="HDBSCAN Clusters")

# Define color scale
unique_clusters = sorted(gdf['hdbscan_cluster'].dropna().unique())
palette = cm.linear.Set1_09.scale(min(unique_clusters), max(unique_clusters)).to_step(len(unique_clusters))

def style_function(feature):
    cluster = feature['properties'].get('hdbscan_cluster')
    color = palette(cluster) if pd.notna(cluster) else "#cccccc"
    return {
        "fillColor": color,
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.7,
    }

# Add GeoJson layer
folium.GeoJson(
    gdf,
    style_function=style_function,
    tooltip=GeoJsonTooltip(fields=["Neighborhood", "hdbscan_cluster"],
                           aliases=["Neighborhood:", "HDBSCAN Cluster:"],
                           localize=True)
).add_to(cluster_layer)

cluster_layer.add_to(m)
folium.LayerControl().add_to(m)

# Save to file
os.makedirs("output", exist_ok=True)
m.save("output/hdbscan_cluster_map.html")
print("✅ HDBSCAN cluster map saved to output/hdbscan_cluster_map.html")
