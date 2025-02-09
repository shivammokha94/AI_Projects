import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
path = os.getcwd()
df = pd.read_csv(path + '/autonomy_data_analysis/data/synthetic_gps_data.csv')

# Convert timestamp column to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Display basic info
print(df.head())
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

## Visualize the Vehicle's Route on a Map
# plt.figure(figsize=(10, 6))
# plt.plot(df["longitude"], df["latitude"], marker="o", linestyle="-", color="b")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Vehicle GPS Path")
# plt.grid()
# plt.show()

## Analyze Speed & Acceleration Over Time
## Speed
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["speed_kmh"], marker="o", linestyle="-", color="g")
plt.xlabel("Timestamp")
plt.ylabel("Speed (km/h)")
plt.title("Speed Over Time")
plt.grid()
plt.xticks(rotation=45)
plt.show()

## Acceleration
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["acceleration_ms2"], marker="s", linestyle="-", color="r")
plt.xlabel("Timestamp")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration Over Time")
plt.grid()
plt.xticks(rotation=45)
plt.show()
