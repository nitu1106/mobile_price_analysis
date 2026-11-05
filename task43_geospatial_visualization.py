# TASK 1: Geospatial Visualization

#
# NOTE: The standard Kaggle Mobile Price Classification dataset does not
# contain latitude/longitude. If coordinates are missing, this script makes
# clearly labelled DEMO city data so you can run and understand the maps.
# Demo locations are NOT real mobile sales/customer locations.

import os
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap

OUTPUT = "task43_output"
os.makedirs(OUTPUT, exist_ok=True)


if os.path.exists("mobile_price.csv"):
    mobile_df = pd.read_csv("mobile_price.csv")
elif os.path.exists("train.csv"):
    mobile_df = pd.read_csv("train.csv")
else:
    mobile_df = pd.DataFrame()
    print("Mobile CSV not found. Demo locations will be used.")


column_names = {str(c).lower(): c for c in mobile_df.columns}
latitude_col = column_names.get("latitude", column_names.get("lat"))
longitude_col = column_names.get(
    "longitude", column_names.get("lon", column_names.get("lng"))
)

if latitude_col and longitude_col:
    map_df = mobile_df.copy()
    map_df["latitude"] = pd.to_numeric(map_df[latitude_col], errors="coerce")
    map_df["longitude"] = pd.to_numeric(map_df[longitude_col], errors="coerce")
    map_df = map_df.dropna(subset=["latitude", "longitude"])
    print("Using coordinates found in the dataset.")
else:
    # Example locations only, to demonstrate how geospatial plots work.
    print("No location columns found. Creating illustrative demo locations.")
    map_df = pd.DataFrame({
        "city": [
            "Ahmedabad", "Mumbai", "Delhi", "Surat", "Jaipur",
            "Bengaluru", "Hyderabad", "Kolkata", "Chennai", "Pune"
        ],
        "country": ["India"] * 10,
        "latitude": [
            23.0225, 19.0760, 28.6139, 21.1702, 26.9124,
            12.9716, 17.3850, 22.5726, 13.0827, 18.5204
        ],
        "longitude": [
            72.5714, 72.8777, 77.2090, 72.8311, 75.7873,
            77.5946, 78.4867, 88.3639, 80.2707, 73.8567
        ],
        "price_range": [0, 1, 2, 3, 1, 2, 3, 0, 1, 2],
        "mobile_count": [120, 95, 110, 80, 70, 105, 90, 75, 85, 65]
    })
    map_df.to_csv(os.path.join(OUTPUT, "DEMO_location_data.csv"), index=False)

# Step 3: Validate coordinates
map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
map_df = map_df.dropna(subset=["latitude", "longitude"])
map_df = map_df[
    map_df["latitude"].between(-90, 90)
    & map_df["longitude"].between(-180, 180)
]

if map_df.empty:
    raise ValueError("No valid latitude and longitude values were found.")

# Step 4: Scatter map using Plotly
fig_scatter = px.scatter_geo(
    map_df,
    lat="latitude",
    lon="longitude",
    color="price_range" if "price_range" in map_df.columns else None,
    hover_name="city" if "city" in map_df.columns else None,
    title="Mobile Data - Location Scatter Map"
)
fig_scatter.update_geos(showland=True, showcountries=True)
fig_scatter.write_html(os.path.join(OUTPUT, "scatter_map.html"))
fig_scatter.show()

# Step 5: Folium heatmap
center = [map_df["latitude"].mean(), map_df["longitude"].mean()]
heat_map = folium.Map(location=center, zoom_start=5, tiles=None)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri",
    name="Esri World Street Map"
).add_to(heat_map)
HeatMap(
    map_df[["latitude", "longitude"]].values.tolist(),
    radius=18,
    blur=15
).add_to(heat_map)
heat_map.save(os.path.join(OUTPUT, "location_heatmap.html"))

# Step 6: Folium marker map
marker_map = folium.Map(location=center, zoom_start=5, tiles=None)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri",
    name="Esri World Street Map"
).add_to(marker_map)
for _, row in map_df.iterrows():
    city = row.get("city", "Location")
    price = row.get("price_range", "Not available")
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"{city} | Price range: {price}"
    ).add_to(marker_map)
marker_map.save(os.path.join(OUTPUT, "location_markers.html"))

# Step 7: Country choropleth (requires country column)
if "country" in map_df.columns:
    if "mobile_count" not in map_df.columns:
        map_df["mobile_count"] = 1

    country_data = map_df.groupby("country", as_index=False)["mobile_count"].sum()
    fig_choropleth = px.choropleth(
        country_data,
        locations="country",
        locationmode="country names",
        color="mobile_count",
        hover_name="country",
        title="Mobile Records by Country"
    )
    fig_choropleth.write_html(os.path.join(OUTPUT, "country_choropleth.html"))
    fig_choropleth.show()
else:
    print("Choropleth skipped: dataset needs a country column.")

print("\nMaps saved in:", OUTPUT)
print("Open the HTML files in a browser to view interactive maps.")
print("IMPORTANT: Demo maps do not show real mobile-user trends.")
