from meteostat import Daily, Stations
from datetime import datetime
import pandas as pd

# === Configuration ===
START_DATE = datetime(2023, 4, 1)
END_DATE = datetime(2025, 4, 1)
OUTPUT_CSV = "seattle_weather_apr2023_apr2025.csv"
LATITUDE = 47.5374
LONGITUDE = -122.3026

# === Step 1: Find nearest weather station (KBFI is Boeing Field) ===
stations = Stations()
station = stations.nearby(LATITUDE, LONGITUDE).fetch(1)
station_id = station.index[0]
print(f"Using station: {station_id}")

# === Step 2: Fetch daily weather data ===
weather_data = Daily(station_id, START_DATE, END_DATE)
df_weather = weather_data.fetch()

# === Step 3: Clean and export ===
df_weather.reset_index(inplace=True)  # Make 'time' a column instead of index
df_weather.rename(columns={"time": "date"}, inplace=True)
df_weather['date'] = pd.to_datetime(df_weather['date']).dt.date  # Keep just date

# Preview
print(df_weather.head())

# Save to CSV
df_weather.to_csv(OUTPUT_CSV, index=False)
print(f"Weather data saved to {OUTPUT_CSV}")
