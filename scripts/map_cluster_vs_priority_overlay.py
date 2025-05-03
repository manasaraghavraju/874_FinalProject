
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from folium.features import GeoJsonTooltip
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Load Data ===
df = pd.read_csv("data/processed/merged_spd_weather.csv", low_memory=False)
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()
valid_df = df[~df['Neighborhood'].isin(['-', 'unknown']) & df['Neighborhood'].notna()]
gdf = gpd.read_file("data/raw/spd_dispatch_neighborhoods.geojson")
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf['Neighborhood'] = gdf['Neighborhood'].str.lower().str.strip()

# === Load Clustering Info ===
cluster_df = pd.read_csv("output/neighborhood_calltype_clusters.csv")
cluster_df["Neighborhood"] = cluster_df["Neighborhood"].str.lower().str.strip()
gdf = gdf.merge(cluster_df[["Neighborhood", "call_type_cluster"]], on="Neighborhood", how="left")

# === Create Map ===
m = folium.Map(location=[47.6, -122.33], zoom_start=12, tiles="CartoDB positron")

# === Draw Cluster Regions ===
n_clusters = cluster_df["call_type_cluster"].nunique()
colormap = cm.get_cmap("Set2", n_clusters)
colors = [mcolors.to_hex(colormap(i)) for i in range(n_clusters)]
color_dict = {i: colors[i] for i in range(n_clusters)}

def style_function(feature):
    cluster = feature["properties"].get("call_type_cluster")
    color = color_dict.get(cluster, "#cccccc") if pd.notna(cluster) else "#cccccc"
    return {
        "fillColor": color,
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.5,
    }

folium.GeoJson(
    gdf,
    tooltip=GeoJsonTooltip(fields=["Neighborhood", "call_type_cluster"],
                           aliases=["Neighborhood:", "Cluster:"],
                           localize=True),
    style_function=style_function,
    name="Call Type Clusters"
).add_to(m)

# === Generate Priority Weighted Heatmap ===
gdf['centroid'] = gdf['geometry'].centroid
centroid_map = dict(zip(gdf['Neighborhood'], gdf['centroid']))

def compute_weight(priority):
    try:
        val = int(priority)
        return max(1, 5 - val)
    except:
        return 1

heat_coords_weighted = []
for _, row in valid_df.iterrows():
    name = row['Neighborhood']
    if name in centroid_map:
        pt = centroid_map[name]
        weight = compute_weight(row.get('Initial Call Priority'))
        heat_coords_weighted.append([pt.y, pt.x, weight])

priority_heat_layer = folium.FeatureGroup(name="High-Priority Call Density")
HeatMap(heat_coords_weighted, radius=10, blur=15, min_opacity=0.3, max_val=4).add_to(priority_heat_layer)
priority_heat_layer.add_to(m)

# === Finalize ===
folium.LayerControl().add_to(m)
m.save("output/cluster_vs_priority_overlay_map.html")
print("✅ Map saved to output/cluster_vs_priority_overlay_map.html")
