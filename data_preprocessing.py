import os
import re  # Added for regular expressions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    return data


def generate_sequences(data, n_past=60):
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def extract_date_features(data):
    # Create a deep copy of the data to prevent SettingWithCopyWarning
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

    # Drop the original 'Date' column
    data_copy = data_copy.drop(columns=['Date'])

    return data_copy

def save_data(data, file_name, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    data.to_csv(os.path.join(folder_name, file_name), index=False)


# Updated function
def standardize_name(name):
    """
    Standardize commodity name by removing special characters.
    """
    cleaned_name = re.sub(r"[^\w\s]", '', name)  # Remove special characters
    cleaned_name = cleaned_name.replace(" ", "_").lower()  # Replace spaces with underscores and make lowercase
    return cleaned_name


def data_preparation(data, target_col_name, n_past=60, skip_feature_extraction_for=None):
    if skip_feature_extraction_for is not None and data['Commodity'].iloc[0] in skip_feature_extraction_for:
        print(f"Skipping feature extraction for {data['Commodity'].iloc[0]}")
    else:
        data = extract_date_features(data)

    encoder = None

    if 'Unit' in data.columns:
        print("Unique values in 'Unit' before encoding:")
        print(data['Unit'].unique())  # Modified line here

        encoder = OneHotEncoder(drop='first', sparse_output=True)
        unit_encoded = encoder.fit_transform(data[['Unit']]).toarray()

        print(f"Shape of unit_encoded: {unit_encoded.shape}")
        print(f"Type of unit_encoded: {type(unit_encoded)}")

        if unit_encoded.shape[1] == 0:
            print("Warning: unit_encoded has no columns after encoding, dropping 'Unit'")
            data = data.drop(columns=['Unit'])
        else:
            if len(encoder.categories_[0]) > 1:
                unit_df = pd.DataFrame(unit_encoded, columns=[f"Unit_{cat}" for cat in encoder.categories_[0][1:]])
            else:
                unit_df = pd.DataFrame(unit_encoded, columns=[f"Unit_{encoder.categories_[0][0]}"])

            data = pd.concat([data, unit_df], axis=1)
            data = data.drop(columns=['Unit'])

    cols = [target_col_name] + [col for col in data if col != target_col_name]
    data = data[cols]

    non_numeric_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found: {non_numeric_cols}. These will be dropped.")
        data = data.drop(columns=non_numeric_cols)

    if data.isnull().values.any() or np.isinf(data.values).any():
        print(f"Warning: NaN or infinite values found in data for {data['Commodity'].iloc[0]}!")

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = generate_sequences(data_scaled, n_past)

    # Check if sequences were generated, if not, return empty arrays and None objects
    if X.size == 0 or y.size == 0:
        print(f"No sequences generated. Skipping further processing...")
        return np.array([]), np.array([]), np.array([]), np.array([]), None, None

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
        plt.plot(X_train[i, :, feature_idx], label=f'Sequence {i+1}')
    plt.title('Sample Sequences of Average Price')
    plt.xlabel('Time Step within Sequence')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()


# Updated 'main()' function
def main():
    file_path = "dataset.csv"
    data = load_data(file_path)
    max_visualizations = 2
    visualized_commodities = 0

    for dir_name in ['models', 'processed_data', 'scalers_encoders']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    for commodity in data['Commodity'].unique():
        print(f"\nProcessing for: {commodity}")
        commodity_data = data[data['Commodity'] == commodity]

        # Standardize file names
        standardized_commodity_name = standardize_name(commodity)

        X_train, X_test, y_train, y_test, scaler, encoder = data_preparation(
            commodity_data,
            target_col_name='Maximum',
            n_past=60,
            skip_feature_extraction_for=['maize']
        )

        # Check if generated sequences are empty
        if X_train.size == 0 or X_test.size == 0:
            print(f"No sequences generated for {commodity}. Skipping...")
            continue

        save_data(pd.DataFrame(X_train.reshape(X_train.shape[0], -1)),
                  f"{standardized_commodity_name}_X_train.csv", 'processed_data')
        save_data(pd.DataFrame(X_test.reshape(X_test.shape[0], -1)),
                  f"{standardized_commodity_name}_X_test.csv", 'processed_data')
        save_data(pd.DataFrame(y_train),
                  f"{standardized_commodity_name}_y_train.csv", 'processed_data')
        save_data(pd.DataFrame(y_test),
                  f"{standardized_commodity_name}_y_test.csv", 'processed_data')

        # Save models
        dump(scaler, f'scalers_encoders/{standardized_commodity_name}_scaler.gz', compress='gzip')
        if encoder is not None:
            dump(encoder, f'scalers_encoders/{standardized_commodity_name}_encoder.gz', compress='gzip')

        if visualized_commodities < max_visualizations:
            print(f"\nVisualizing for: {commodity}")
            plot_training_target_distribution(y_train, y_test)
            plot_average_price_sequence(X_train)
            visualized_commodities += 1
        else:
            print(f"\nSkipping visualization for: {commodity}")

        print(f"Summary for {commodity}:")
        print(f" - Training data: {X_train.shape[0]} sequences")
        print(f" - Test data: {X_test.shape[0]} sequences\n")
        print("-" * 30)


if __name__ == "__main__":
    main()
