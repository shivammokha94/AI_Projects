import os, sys
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the dataset
path = os.getcwd()
df = pd.read_csv(path + '/autonomy_data_analysis/data/synthetic_gps_data.csv')

# Convert timestamp to index
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Train ARIMA model
model = ARIMA(df["speed_kmh"], order=(5,1,0))  # (p,d,q)
model_fit = model.fit()

# Forecast next 10 seconds
forecast = model_fit.forecast(steps=10)
print(forecast)

# Plot the prediction
plt.figure(figsize=(10, 5))
plt.plot(df.index[-50:], df["speed_kmh"].iloc[-50:], label="Actual Speed", color="blue")
plt.plot(pd.date_range(df.index[-1], periods=10, freq="1S"), forecast, label="Predicted Speed", color="red")
plt.legend()
plt.title("Speed Prediction Using ARIMA")
plt.show()
