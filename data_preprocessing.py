import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    return data


# Function to generate sequences
def generate_sequences(data, n_past=60):
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# Function to extract date features
def extract_date_features(data):
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    data['Month_Sin'] = np.sin((data['Month'] - 1) * (2. * np.pi / 12))
    data['Month_Cos'] = np.cos((data['Month'] - 1) * (2. * np.pi / 12))
    data['Day_Sin'] = np.sin((data['Day'] - 1) * (2. * np.pi / 30))
    data['Day_Cos'] = np.cos((data['Day'] - 1) * (2. * np.pi / 30))
    data['DayOfWeek_Sin'] = np.sin(data['DayOfWeek'] * (2. * np.pi / 7))
    data['DayOfWeek_Cos'] = np.cos(data['DayOfWeek'] * (2. * np.pi / 7))

    data = data.drop(columns=['Date'])
    return data


# Updated data preparation function
def data_preparation(data, target_col_name, n_past=60):
    # Drop the 'Commodity' column
    data = data.drop(columns=['Commodity'])

    # Extract date features
    data = extract_date_features(data)

    # Arrange the 'Average' column to be the first
    cols = [target_col_name] + [col for col in data if col != target_col_name]
    data = data[cols]

    # Check for non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found: {non_numeric_cols}. These will be dropped.")
        data = data.drop(columns=non_numeric_cols)

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Generate sequences
    X, y = generate_sequences(data_scaled, n_past)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler


# Visualization Function 1: Plotting Distribution
def plot_training_target_distribution(y_train, y_test):
    plt.figure(figsize=(10, 5))
    plt.hist(y_train, bins=30, alpha=0.5, label='Training')
    plt.hist(y_test, bins=30, alpha=0.5, label='Testing')
    plt.title('Distribution of Target Variable in Training and Testing Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Visualization Function 2: Sample Sequences
def plot_average_price_sequence(X_train, feature_idx=0, num_sequences=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_sequences):
        plt.plot(X_train[i, :, feature_idx], label=f'Sequence {i+1}')
    plt.title('Sample Sequences of Average Price')
    plt.xlabel('Time Step within Sequence')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    file_path = "dataset.csv"
    data = load_data(file_path)

    commodity_data = data[data['Commodity'] == "Tomato Big(Nepali)"]

    X_train, X_test, y_train, y_test, scaler = data_preparation(
        commodity_data,
        target_col_name='Average',
        n_past=60
    )

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Calling Visualization Functions
    plot_training_target_distribution(y_train, y_test)
    plot_average_price_sequence(X_train)