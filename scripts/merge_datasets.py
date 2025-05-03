import pandas as pd

# === CONFIGURATION ===
SPD_CALLS_PATH = "data/raw/SeattlePD_CallDataset.csv"  # or full dataset CSV
WEATHER_DATA_PATH = "data/raw/seattle_weather_apr2023_apr2025.csv"
OUTPUT_PATH = "data/processed/merged_spd_weather.csv"
CALL_TIMESTAMP_COL = "CAD Event Original Time Queued"

# === STEP 1: Load datasets ===
print("Loading datasets...")
calls_df = pd.read_csv(SPD_CALLS_PATH)
weather_df = pd.read_csv(WEATHER_DATA_PATH)

# === STEP 2: Convert dates ===
print("Converting date columns...")
calls_df[CALL_TIMESTAMP_COL] = pd.to_datetime(calls_df[CALL_TIMESTAMP_COL], errors='coerce')
calls_df['date'] = calls_df[CALL_TIMESTAMP_COL].dt.date
calls_df['date'] = pd.to_datetime(calls_df['date'])

weather_df['date'] = pd.to_datetime(weather_df['date'])

# === STEP 3: Merge on 'date' ===
print("Merging datasets...")
merged_df = pd.merge(calls_df, weather_df, on='date', how='left')

# === STEP 4: Save to file ===
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"Merged dataset saved to {OUTPUT_PATH}")
