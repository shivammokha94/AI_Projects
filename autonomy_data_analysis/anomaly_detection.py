import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px

# Load the dataset
path = os.getcwd()
df = pd.read_csv(path + '/autonomy_data_analysis/data/synthetic_gps_data.csv')

# Select features for anomaly detection
features = ["speed_kmh", "acceleration_ms2"]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Train Isolation Forest (unsupervised anomaly detection)
model = IsolationForest(contamination=0.10, random_state=42)
df["anomaly"] = model.fit_predict(df_scaled)

# Mark anomalies (-1 = anomaly, 1 = normal)
df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

# Count anomalies
print(df["anomaly"].value_counts())

fig = px.scatter(df, x="speed_kmh", y="acceleration_ms2", color="anomaly",
                 title="Anomaly Detection in Driving Behavior")
fig.show()

