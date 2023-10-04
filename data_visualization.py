import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Load Data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    return data


# Preprocess Data
def preprocess_data(data, target_col_name, n_past=60):
    data = data.drop(columns=['Commodity', 'Unit'])  # dropping irrelevant columns

    # Arrange the target column to be the first
    cols = [target_col_name] + [col for col in data if col != target_col_name]
    data_feature = data[cols].values

    # Scaling the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_feature)

    # Generating sequences
    X, y = [], []
    for i in range(n_past, len(data_scaled)):
        X.append(data_scaled[i - n_past:i, :])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Splitting data without shuffling (important for time-series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler


# Visualizations
def plot_training_target_distribution(y_train, y_test):
    plt.figure(figsize=(10, 5))
    plt.hist(y_train, bins=30, alpha=0.5, label='Training')
    plt.hist(y_test, bins=30, alpha=0.5, label='Testing')
    plt.title('Distribution of Target Variable in Training and Testing Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def plot_average_price_sequence(X_train, feature_idx=0, num_sequences=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_sequences):
        plt.plot(X_train[i, :, feature_idx], label=f'Sequence {i + 1}')
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

    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        commodity_data,
        target_col_name='Average',
        n_past=60
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Uncomment the following lines when you want to visualize the data
    plot_training_target_distribution(y_train, y_test)
    plot_average_price_sequence(X_train)
