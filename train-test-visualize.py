import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

# Folder paths
data_folder = "processed_data/"
scaler_folder = "scalers_encoders/"

# Fetching all file names
all_files = os.listdir(data_folder)
unique_prefixes = set(fname.split('_')[0] for fname in all_files)

for prefix in unique_prefixes:
    try:
        # Load Data
        X_train = pd.read_csv(data_folder + prefix + '_X_train.csv')
        y_train = pd.read_csv(data_folder + prefix + '_y_train.csv')
        X_test = pd.read_csv(data_folder + prefix + '_X_test.csv')
        y_test = pd.read_csv(data_folder + prefix + '_y_test.csv')

        # Load Scaler
        scaler = joblib.load(scaler_folder + prefix + '_scaler.gz')

        # Reshape X for LSTM
        X_train_reshaped = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

        # Define LSTM Model
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Train the Model
        model.fit(
            X_train_reshaped, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_reshaped, y_test),
            verbose=2,
            shuffle=False
        )

        # Save the Model
        model.save(prefix + "_model.h5")

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}. Skipping {prefix} and moving to the next prefix.")
        continue  # Continue to the next prefix if a file is not found

    except Exception as e:
        print(f"An error occurred: {str(e)}. Skipping {prefix} and moving to the next prefix.")
        continue

