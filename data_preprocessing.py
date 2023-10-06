import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load
import warnings  # Import warnings module

# Suppress specific warning
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


def save_data(data, file_name):
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    data.to_csv(os.path.join('processed_data', file_name), index=False)


def data_preparation(data, target_col_name, n_past=60, skip_feature_extraction_for=None):
    if skip_feature_extraction_for is not None and data['Commodity'].iloc[0] in skip_feature_extraction_for:
        print(f"Skipping feature extraction for {data['Commodity'].iloc[0]}")
    else:
        data = extract_date_features(data)

    encoder = None

    if 'Unit' in data.columns:
        encoder = OneHotEncoder(drop='first', sparse_output=True)  # Updated based on warning
        unit_encoded = encoder.fit_transform(data[['Unit']]).toarray()  # Convert to dense array

        # Debugging: Check the shape and type
        print(f"Shape of unit_encoded: {unit_encoded.shape}")
        print(f"Type of unit_encoded: {type(unit_encoded)}")

        if unit_encoded.shape[1] == 0:  # Check if there are no columns
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
        # Consider handling or skipping this commodity...

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
        plt.plot(X_train[i, :, feature_idx], label=f'Sequence {i+1}')
    plt.title('Sample Sequences of Average Price')
    plt.xlabel('Time Step within Sequence')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()


# Main function
if __name__ == "__main__":
    file_path = "dataset.csv"
    data = load_data(file_path)

    max_visualizations = 5
    visualized_commodities = 0

    if not os.path.exists('models'):
        os.makedirs('models')

    for commodity in data['Commodity'].unique():
        if commodity == 'Maize':  # Skip all processing for Maize
            print("Skipping all processing for Maize")
            continue

        print(f"\nProcessing for: {commodity}")
        commodity_data = data[data['Commodity'] == commodity]

        X_train, X_test, y_train, y_test, scaler, encoder = data_preparation(
            commodity_data,
            target_col_name='Average',
            n_past=60,
            skip_feature_extraction_for=['Maize']
        )

        save_data(pd.DataFrame(X_train.reshape(X_train.shape[0], -1)), f"{commodity}_X_train.csv")
        save_data(pd.DataFrame(X_test.reshape(X_test.shape[0], -1)), f"{commodity}_X_test.csv")
        save_data(pd.DataFrame(y_train), f"{commodity}_y_train.csv")
        save_data(pd.DataFrame(y_test), f"{commodity}_y_test.csv")

        if not os.path.exists('scalers_encoders'):
            os.makedirs('scalers_encoders')

        valid_filename_commodity = commodity.replace("(", "_").replace(")", "_").replace(" ", "_")
        dump(scaler, f'scalers_encoders/{valid_filename_commodity}_scaler.gz', compress='gzip')
        if encoder is not None:
            dump(encoder, f'scalers_encoders/{valid_filename_commodity}_encoder.gz', compress='gzip')

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


