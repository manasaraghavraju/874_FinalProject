
import pandas as pd
import geopandas as gpd
from difflib import get_close_matches
import folium
from folium.features import GeoJsonTooltip
import branca.colormap as cm

# === Load SPD Neighborhood GeoJSON ===
gdf = gpd.read_file("data/raw/spd_dispatch_neighborhoods.geojson")
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf["Neighborhood"] = gdf["Neighborhood"].str.lower().str.strip()

# === Load Population GeoJSON ===
pop_gdf = gpd.read_file("data/raw/hh_population_types_Neighborhoods_5617280960769611352.geojson")
pop_gdf["Original_Pop_Name"] = pop_gdf["NEIGH_NAME"]
pop_gdf["Neighborhood"] = pop_gdf["NEIGH_NAME"].str.lower().str.strip()

# === Fuzzy Matching ===
spd_names = set(gdf["Neighborhood"].dropna())
pop_names = set(pop_gdf["Neighborhood"].dropna())

fuzzy_mapping = {}
for name in spd_names:
    match = get_close_matches(name, pop_names, n=1, cutoff=0.6)
    if match:
        fuzzy_mapping[name] = match[0]

# Reverse mapping and filter pop_gdf to only matched
reverse_map = {v: k for k, v in fuzzy_mapping.items()}
pop_gdf = pop_gdf[pop_gdf["Neighborhood"].isin(reverse_map)]
pop_gdf["Neighborhood"] = pop_gdf["Neighborhood"].map(reverse_map)

# Prepare population dataframe
pop_df = pop_gdf[["Neighborhood", "TOTAL_POPULATION"]].copy()
pop_df["TOTAL_POPULATION"] = pd.to_numeric(pop_df["TOTAL_POPULATION"], errors="coerce")

# === Merge with SPD Neighborhoods ===
gdf = gdf.merge(pop_df, on="Neighborhood", how="left")

# === Compute Area and Density ===
gdf = gdf.to_crs(epsg=3395)  # Project to metric CRS
gdf["area_km2"] = gdf.geometry.area / 1e6
gdf["population_density"] = gdf["TOTAL_POPULATION"] / gdf["area_km2"]

# === Create Map ===
gdf = gdf.to_crs(epsg=4326)  # Convert back to lat/lon
m = folium.Map(location=[47.6, -122.33], zoom_start=12, tiles="CartoDB positron")

# Color scale
min_density = gdf["population_density"].min()
max_density = gdf["population_density"].max()
color_scale = cm.linear.YlGnBu_09.scale(min_density, max_density)
color_scale.caption = "Population Density (people/km²)"
color_scale.add_to(m)

# Safe styling function
def style_density(feature):
    val = feature["properties"].get("population_density")
    if val is None or pd.isna(val):
        return {
            "fillColor": "#cccccc",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5,
        }
    return {
        "fillColor": color_scale(val),
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.7,
    }

# Add GeoJson layer
folium.GeoJson(
    gdf,
    tooltip=GeoJsonTooltip(fields=["Neighborhood", "TOTAL_POPULATION", "population_density"],
                           aliases=["Neighborhood:", "Population:", "Pop. Density (per km²):"],
                           localize=True),
    style_function=style_density,
    name="Population Density"
).add_to(m)

folium.LayerControl().add_to(m)
m.save("interactive_population_density_map.html")
print("✅ Map saved to interactive_population_density_map.html")
