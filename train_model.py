import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import data_preparation, load_data
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and prepare data
file_path = "dataset.csv"
data = load_data(file_path)
commodity_data = data[data['Commodity'] == "Tomato Big(Nepali)"]
X_train, X_test, y_train, y_test, scaler = data_preparation(commodity_data, target_col_name='Average', n_past=60)

# Model building
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()

# Directory to save the model
model_dir = './model'
os.makedirs(model_dir, exist_ok=True)

# Callbacks
checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), save_best_only=True, monitor='val_loss', mode='min')

# Model training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint],
    shuffle=False
)

# Save the final model
model.save(os.path.join(model_dir, 'final_model.h5'))
