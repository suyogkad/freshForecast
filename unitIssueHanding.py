import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load


def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    return data


def generate_sequences(data, n_past=60):
    X, y = [], []
    for i in range(n_past, len(data)):
        X_sequence = data[i - n_past:i, :]
        y_value = data[i, 0]

        # Check for NaN in the sequences and target and skip if found
        if not np.isnan(X_sequence).any() and not np.isnan(y_value):
            X.append(X_sequence)
            y.append(y_value)

    return np.array(X), np.array(y)


def data_preparation(data, target_col_name, n_past=60):
    data = extract_date_features(data)
    encoder = None
    if 'Unit' in data.columns:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        unit_encoded = encoder.fit_transform(data[['Unit']])
        unit_df = pd.DataFrame(unit_encoded, columns=[f"Unit_{cat}" for cat in encoder.categories_[0][1:]])
        data = pd.concat([data, unit_df], axis=1)
        data = data.drop(columns=['Unit'])

    cols = [target_col_name] + [col for col in data if col != target_col_name]
    data = data[cols]

    # Check NaN after encoding and concatenating
    print(f"NaN after encoding and concatenating: {data.isna().sum().sum()}")

    non_numeric_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found: {non_numeric_cols}. These will be dropped.")
        data = data.drop(columns=non_numeric_cols)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Check NaN after scaling
    print(f"NaN after scaling: {np.isnan(data_scaled).sum()}")

    X, y = generate_sequences(data_scaled, n_past)

    # Check NaN after sequence generation
    print(f"NaN in X after sequence generation: {np.isnan(X).sum()}")
    print(f"NaN in y after sequence generation: {np.isnan(y).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler, encoder


def extract_date_features(data):
    data_copy = data.copy(deep=True)
    data_copy['Year'] = data_copy['Date'].dt.year
    data_copy['Month'] = data_copy['Date'].dt.month
    data_copy['Day'] = data_copy['Date'].dt.day
    data_copy['DayOfWeek'] = data_copy['Date'].dt.dayofweek
    data_copy['Month_Sin'] = np.sin((data_copy['Month'] - 1) * (2. * np.pi / 12))
    data_copy['Month_Cos'] = np.cos((data_copy['Month'] - 1) * (2. * np.pi / 12))
    data_copy['Day_Sin'] = np.sin((data_copy['Day'] - 1) * (2. * np.pi / 30))
    data_copy['Day_Cos'] = np.cos((data_copy['Day'] - 1) * (2. * np.pi / 30))
    data_copy['DayOfWeek_Sin'] = np.sin(data_copy['DayOfWeek'] * (2. * np.pi / 7))
    data_copy['DayOfWeek_Cos'] = np.cos(data_copy['DayOfWeek'] * (2. * np.pi / 7))
    data_copy = data_copy.drop(columns=['Date'])
    return data_copy


def save_data(data, file_name):
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    data.to_csv(os.path.join('processed_data', file_name), index=False)


def data_preparation(data, target_col_name, n_past=60):
    data = extract_date_features(data)
    encoder = None

    if 'Unit' in data.columns:
        encoder = OneHotEncoder(drop='first', sparse=False)
        unit_encoded = encoder.fit_transform(data[['Unit']])
        unit_df = pd.DataFrame(unit_encoded, columns=[f"Unit_{cat}" for cat in encoder.categories_[0][1:]])
        data = pd.concat([data, unit_df], axis=1)
        data = data.drop(columns=['Unit'])

    cols = [target_col_name] + [col for col in data if col != target_col_name]
    data = data[cols]
    non_numeric_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found: {non_numeric_cols}. These will be dropped.")
        data = data.drop(columns=non_numeric_cols)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = generate_sequences(data_scaled, n_past)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler, encoder


def plot_training_target_distribution(y_train, y_test):
    if np.isnan(y_train).any() or np.isnan(y_test).any():
        print("Warning: NaN values found in the data. Removing them for visualization.")
        y_train = y_train[~np.isnan(y_train)]
        y_test = y_test[~np.isnan(y_test)]

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


if __name__ == "__main__":
    file_path = "dataset.csv"
    data = load_data(file_path)

    # Filter data to work with Maize only
    maize_data = data[data['Commodity'] == 'Maize']

    # Troubleshooting Step 1: Check basic data info and null values
    print("Info of Maize Data:")
    print(maize_data.info())
    print("\nNull Values in Maize Data:")
    print(maize_data.isnull().sum())
    print("\nDescriptive Stats for Maize Data:")
    print(maize_data.describe())

    # Troubleshooting Step 2: Check the uniqueness and consistency of categorical columns
    print("\nUnique Values in 'Commodity' Column:")
    print(maize_data['Commodity'].unique())

    # Additional troubleshooting could involve visualizing certain columns
    # Troubleshooting Step 3: Visualize the data
    plt.figure(figsize=(10, 6))
    plt.plot(maize_data['Date'], maize_data['Average'])
    plt.title("Average Price of Maize Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.show()

    # Data preparation - same steps as before but added prints to troubleshoot
    # where it might break or produce unexpected results
    print("\nData Preparation for Maize:")

    n_past = min(60, len(maize_data)-1)  # Adjust n_past to not exceed available data points

    try:
        X_train, X_test, y_train, y_test, scaler, encoder = data_preparation(
            maize_data,
            target_col_name='Average',
            n_past=n_past  # Use adjusted n_past
        )
    except Exception as e:
        print(f"An error occurred during data preparation: {str(e)}")

    # Check shapes of training/testing arrays to ensure correctness and non-nan values
    print(
        f"\nShapes: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(
        f"NaN Check: X_train: {np.isnan(X_train).any()}, X_test: {np.isnan(X_test).any()}, y_train: {np.isnan(y_train).any()}, y_test: {np.isnan(y_test).any()}")

    # Saving, Visualization, and Summary as per your original script but for Maize only.
    # The following parts of the script assume that the above steps run error-free and
    # may need to be adjusted based on the results of the troubleshooting steps.

    # Saving data
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

    save_data(pd.DataFrame(X_train.reshape(X_train.shape[0], -1)), "Maize_X_train.csv")
    save_data(pd.DataFrame(X_test.reshape(X_test.shape[0], -1)), "Maize_X_test.csv")
    save_data(pd.DataFrame(y_train), "Maize_y_train.csv")
    save_data(pd.DataFrame(y_test), "Maize_y_test.csv")

    # Save the scaler and encoder objects for future use
    if not os.path.exists('models'):
        os.makedirs('models')

    dump(scaler, 'models/Maize_scaler.gz', compress='gzip')
    if encoder is not None:
        dump(encoder, 'models/Maize_encoder.gz', compress='gzip')

    # Visualization
    print("\nVisualizing for: Maize")
    plot_training_target_distribution(y_train, y_test)
    plot_average_price_sequence(X_train)

    # Summary
    print(f"\nSummary for Maize:")
    print(f" - Training data: {X_train.shape[0]} sequences")
    print(f" - Test data: {X_test.shape[0]} sequences\n")
    print("-" * 30)
