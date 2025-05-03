
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Load Data ===
df = pd.read_csv("output/neighborhood_calltype_clusters.csv")
gdf = gpd.read_file("data/raw/spd_dispatch_neighborhoods.geojson")
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf["Neighborhood"] = gdf["Neighborhood"].str.lower().str.strip()
df["Neighborhood"] = df["Neighborhood"].str.lower().str.strip()

# === Merge Data ===
gdf_clustered = gdf.merge(df[["Neighborhood", "call_type_cluster", "total_calls"]], on="Neighborhood", how="left")

# === Generate Color Map ===
n_clusters = df["call_type_cluster"].nunique()
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
        "fillOpacity": 0.7,
    }

# === Create Map ===
m = folium.Map(location=[47.6, -122.33], zoom_start=12, tiles="CartoDB positron")

folium.GeoJson(
    gdf_clustered,
    tooltip=GeoJsonTooltip(
        fields=["Neighborhood", "call_type_cluster", "total_calls"],
        aliases=["Neighborhood:", "Cluster:", "Total Calls:"],
        localize=True
    ),
    style_function=style_function,
    name="Call Type Clusters"
).add_to(m)

folium.LayerControl().add_to(m)
m.save("output/call_type_clusters_map.html")
print("✅ Map saved to output/call_type_clusters_map.html")
