
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from folium.features import GeoJsonTooltip
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Load GeoData and Cluster Data ===
df = pd.read_csv("data/processed/merged_spd_weather.csv", low_memory=False)
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()

gdf = gpd.read_file("data/raw/spd_dispatch_neighborhoods.geojson")
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf['Neighborhood'] = gdf['Neighborhood'].str.lower().str.strip()

clusters = pd.read_csv("output/neighborhood_pca_clusters.csv")
clusters["Neighborhood"] = clusters["Neighborhood"].str.lower().str.strip()

# Merge clusters into GeoDataFrame
gdf = gdf.merge(clusters[["Neighborhood", "pca_cluster", "total_calls"]], on="Neighborhood", how="left")
gdf = gdf.to_crs(epsg=4326)

# Create map
m = folium.Map(location=[47.6, -122.33], zoom_start=12, tiles="CartoDB positron")

# Define cluster colors
n_clusters = gdf["pca_cluster"].nunique()
colormap = cm.get_cmap("Set1", n_clusters)
colors = [mcolors.to_hex(colormap(i)) for i in range(n_clusters)]
color_dict = {i: colors[i] for i in range(n_clusters)}

# === Layer 1: PCA Cluster Choropleth ===
folium.GeoJson(
    gdf,
    tooltip=GeoJsonTooltip(fields=["Neighborhood", "pca_cluster", "total_calls"],
                           aliases=["Neighborhood:", "PCA Cluster:", "Total Calls:"],
                           localize=True),
    style_function=lambda feature: {
        "fillColor": color_dict.get(int(feature["properties"]["pca_cluster"]), "#cccccc")
                        if feature["properties"]["pca_cluster"] is not None else "#cccccc",
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.5
    },
    name="PCA Call Type Clusters"
).add_to(m)

# === Layer 2: High-Priority Call Heatmap ===
gdf['centroid'] = gdf['geometry'].centroid
centroid_map = dict(zip(gdf['Neighborhood'], gdf['centroid']))

def compute_weight(priority):
    try:
        val = int(priority)
        return max(1, 5 - val)
    except:
        return 1

df['Neighborhood'] = df['Neighborhood'].astype(str).str.lower().str.strip()
df = df[df['Neighborhood'].isin(centroid_map)]

heat_coords_weighted = []
for _, row in df.iterrows():
    pt = centroid_map[row['Neighborhood']]
    weight = compute_weight(row.get('Initial Call Priority'))
    heat_coords_weighted.append([pt.y, pt.x, weight])

HeatMap(heat_coords_weighted, radius=10, blur=15, min_opacity=0.2, max_val=4).add_to(
    folium.FeatureGroup(name="Heatmap: High-Priority Calls").add_to(m)
)

# Finalize map
folium.LayerControl().add_to(m)
m.save("output/pca_cluster_vs_priority_overlay_map.html")
print("✅ Map saved to output/pca_cluster_vs_priority_overlay_map.html")
