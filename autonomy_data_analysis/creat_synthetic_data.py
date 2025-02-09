#%pip install pandas numpy

import pandas as pd
import numpy as np
import os, sys

# Generate timestamps
time_steps = pd.date_range(start="2024-01-01 12:00:00", periods=100, freq="1s")

# Simulated GPS coordinates (random walk)
latitudes = np.cumsum(np.random.normal(0, 0.0001, 100)) + 37.7749  # Centered near SF
longitudes = np.cumsum(np.random.normal(0, 0.0001, 100)) - 122.4194

# Simulated speed (km/h) and acceleration (m/s²)
speeds = np.abs(np.random.normal(30, 5, 100))  # Avg 30 km/h
accelerations = np.gradient(speeds)  # Rate of change of speed

# Create DataFrame
df = pd.DataFrame({"timestamp": time_steps, "latitude": latitudes, "longitude": longitudes, 
                   "speed_kmh": speeds, "acceleration_ms2": accelerations})

# Save to CSV
path = os.getcwd()
df.to_csv(path + '/autonomy_data_analysis/data/synthetic_gps_data.csv', index=False)

print("Synthetic GPS data generated successfully!")