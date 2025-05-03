
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from folium.features import GeoJsonTooltip
import branca.colormap as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === Load Data ===
df = pd.read_csv("data/processed/merged_spd_weather.csv", low_memory=False)
gdf = gpd.read_file("data/raw/spd_dispatch_neighborhoods.geojson")

# === Normalize Neighborhood Names ===
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf['Neighborhood'] = gdf['Neighborhood'].str.lower().str.strip()
gdf_web = gdf.to_crs(epsg=4326)

df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()

# === Filter Valid Neighborhoods ===
valid_df = df[~df['Neighborhood'].isin(['-', 'unknown']) & df['Neighborhood'].notna()]

# === Call Volume Counts ===
call_counts = valid_df['Neighborhood'].value_counts().reset_index()
call_counts.columns = ['Neighborhood', 'call_count']

# === Merge with GeoDataFrame ===
gdf_web = gdf_web.merge(call_counts, on='Neighborhood', how='left')
if 'call_count' not in gdf_web.columns:
    gdf_web['call_count'] = 0
gdf_web['call_count'] = gdf_web['call_count'].fillna(0)

# === Clustering by Volume ===
if gdf_web['call_count'].nunique() > 1:
    gdf_web['cluster'] = pd.qcut(gdf_web['call_count'], q=4, labels=False, duplicates="drop")
else:
    gdf_web['cluster'] = 0

# === Compute Top Call Types ===
df['Initial Call Type'] = df['Initial Call Type'].astype(str).str.lower().str.strip()
top_calls = (
    df.groupby(['Neighborhood', 'Initial Call Type'])
      .size()
      .reset_index(name='count')
      .sort_values(['Neighborhood', 'count'], ascending=[True, False])
)
top_calls_ranked = top_calls.groupby('Neighborhood').head(3)
call_type_map = (
    top_calls_ranked.groupby('Neighborhood')['Initial Call Type']
    .apply(lambda x: ', '.join(x.head(3)))
    .to_dict()
)
gdf_web['Top Call Types'] = gdf_web['Neighborhood'].map(call_type_map)

# === Call Type Clustering ===
call_type_matrix = (
    valid_df.groupby(['Neighborhood', 'Initial Call Type'])
    .size()
    .unstack(fill_value=0)
)
call_type_dist = call_type_matrix.div(call_type_matrix.sum(axis=1), axis=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(call_type_dist)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
call_type_dist['call_type_cluster'] = kmeans.fit_predict(X_scaled)
call_type_dist['total_calls'] = call_type_matrix.sum(axis=1)
call_type_dist.reset_index().to_csv("output/neighborhood_calltype_clusters.csv", index=False)
print("✅ Saved call type clusters to output/neighborhood_calltype_clusters.csv")

# === Create Base Map ===
m = folium.Map(location=[47.6, -122.33], zoom_start=12, tiles='CartoDB positron')

# === Call Volume Choropleth ===
volume_layer = folium.FeatureGroup(name="Choropleth: Call Volume")
min_calls = gdf_web['call_count'].min()
max_calls = gdf_web['call_count'].max()
color_scale = cm.linear.OrRd_09.scale(min_calls, max_calls)
color_scale.caption = "911 Call Volume"
color_scale.add_to(m)

def style_by_volume(feature):
    call_count = feature['properties']['call_count']
    return {
        'fillColor': color_scale(call_count),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7,
    }

folium.GeoJson(
    gdf_web,
    tooltip=GeoJsonTooltip(fields=['Neighborhood', 'call_count', 'Top Call Types'],
                           aliases=['Neighborhood:', 'Calls:', 'Top Types:'],
                           localize=True),
    style_function=style_by_volume
).add_to(volume_layer)
volume_layer.add_to(m)

# === Heatmap by Call Density ===
gdf_web['centroid'] = gdf_web['geometry'].centroid
centroid_map = dict(zip(gdf_web['Neighborhood'], gdf_web['centroid']))
heat_coords = [
    [centroid_map[name].y, centroid_map[name].x]
    for name in valid_df['Neighborhood']
    if name in centroid_map
]

heat_layer = folium.FeatureGroup(name="Heatmap: Call Density")
HeatMap(heat_coords, radius=8, blur=12, min_opacity=0.2).add_to(heat_layer)
heat_layer.add_to(m)

# === Priority Weighted Heatmap ===
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

priority_heat_layer = folium.FeatureGroup(name="Heatmap: High-Priority Calls")
HeatMap(heat_coords_weighted, radius=10, blur=15, min_opacity=0.2, max_val=4).add_to(priority_heat_layer)
priority_heat_layer.add_to(m)

# === Finalize ===
folium.LayerControl().add_to(m)
m.save("output/interactive_seattle_911_fullmap.html")
print("✅ Map saved to output/interactive_seattle_911_fullmap.html")
