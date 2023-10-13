import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
import joblib

# Create a directory if not exists
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Directories
dataset_dir = 'flask_app'
saved_data_dir = 'saved_data'

create_dir(saved_data_dir)

# Data Preprocessing

# Load Data
data = pd.read_csv(os.path.join(dataset_dir, 'dataset.csv'))

# Filter to only use data where the unit is 'Kg'
data = data[data['Unit'] == 'Kg']

# Extract unique commodity names and save for future use
commodity_names = list(data['Commodity'].unique())
joblib.dump(commodity_names, os.path.join(saved_data_dir, 'commodity_names.pkl'))


# Data Preprocessing
data = pd.read_csv('flask_app/dataset.csv')
data = data[data['Unit'] == 'Kg']
commodity_names = list(data['Commodity'].unique())
joblib.dump(commodity_names, 'saved_data/commodity_names.pkl')
data = pd.get_dummies(data, columns=['Commodity'])
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year.astype(float)
data['Month'] = data['Date'].dt.month.astype(float)
data['Day'] = data['Date'].dt.day.astype(float)
data = data.drop(columns=['Date', 'Unit', 'SN'])
data = data.dropna()
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
target = ['Minimum', 'Maximum']
features = [col for col in data.columns if col not in target]
X = data[features]
y = data[target]
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)
scaler_y = StandardScaler().fit(y)
y_scaled = scaler_y.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Model Building & Training
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(2)
])
model.compile(optimizer='adam', loss='mse')
print("Model Training...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2,
    shuffle=False
)
model.save('lstm_model.h5')
print("Model Saved!")

# Predicting values
y_pred = model.predict(X_test)

# Inverting scaling for prediction and actual values
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

# Calculating Mean Absolute Error, Mean Squared Error, and R2 Score
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Visualizing Model Training History

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save models and scalers
model.save(os.path.join(saved_data_dir, 'lstm_model.h5'))

np.save(os.path.join(saved_data_dir, 'X_train.npy'), X_train)
np.save(os.path.join(saved_data_dir, 'X_test.npy'), X_test)
np.save(os.path.join(saved_data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(saved_data_dir, 'y_test.npy'), y_test)

joblib.dump(scaler_X, os.path.join(saved_data_dir, 'scaler_X.pkl'))
joblib.dump(scaler_y, os.path.join(saved_data_dir, 'scaler_y.pkl'))

print("Model, data, and scalers saved!")
