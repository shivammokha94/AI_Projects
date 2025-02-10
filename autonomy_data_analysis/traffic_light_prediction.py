import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
path = os.getcwd() + '/autonomy_data_analysis/data/'

# Load and preprocess images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory("traffic_light_dataset", target_size=(64, 64), batch_size=32, class_mode="binary")

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")  # Output: Red (1) or Green (0)
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_data, epochs=10)

# Save model
model.save(path + "traffic_light_classifier.h5")
