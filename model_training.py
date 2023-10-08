import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import gzip

# Suppress TensorFlow warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Directories
processed_data_dir = "processed_data"
scalers_encoders_dir = "scalers_encoders"
models_dir = "models"

# Ensure directory exists
os.makedirs(models_dir, exist_ok=True)

def format_commodity_name(commodity):
    """Format commodity name to be used in file paths."""
    return commodity.replace("(", "_").replace(")", "_").replace(" ", "_").replace("/", "_")

def build_and_train_model(X_train, y_train, X_test, y_test):
    # Reshape X_train to be 3D for LSTM
    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

    # Define model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=2,
        shuffle=False
    )
    return model, history

# Extract commodities from filenames
commodities = set(file.replace("_X_train.csv", "") for file in os.listdir(processed_data_dir) if "_X_train.csv" in file)

print("Identified commodities: ", commodities)

for commodity in commodities:
    formatted_commodity_name = format_commodity_name(commodity)

    # Load data
    X_train = pd.read_csv(os.path.join(processed_data_dir, f"{formatted_commodity_name}_X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_data_dir, f"{formatted_commodity_name}_y_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_data_dir, f"{formatted_commodity_name}_X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_data_dir, f"{formatted_commodity_name}_y_test.csv"))

    print(f"\nShape of X_train for {commodity}: {X_train.shape}")

    # Train model
    model, history = build_and_train_model(X_train, y_train, X_test, y_test)

    # Save model
    model_filename = os.path.join(models_dir, f"{formatted_commodity_name}_model.h5")
    model.save(model_filename)

    # Evaluate model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"{commodity} Test RMSE: {rmse:.3f}")
