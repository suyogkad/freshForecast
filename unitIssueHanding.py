import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    return data


def generate_sequences(data, n_past=60):
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def extract_date_features(data):
    data_copy = data.copy(deep=True)
    data_copy['Year'] = data_copy['Date'].dt.year
    data_copy['Month'] = data_copy['Date'].dt.month
    data_copy['Day'] = data_copy['Date'].dt.day
    data_copy['DayOfWeek'] = data_copy['Date'].dt.dayofweek
    data_copy['Month_Sin'] = np.sin(2 * np.pi * data_copy['Month'] / 12)
    data_copy['Month_Cos'] = np.cos(2 * np.pi * data_copy['Month'] / 12)
    data_copy['Day_Sin'] = np.sin(2 * np.pi * data_copy['Day'] / 30)
    data_copy['Day_Cos'] = np.cos(2 * np.pi * data_copy['Day'] / 30)
    data_copy['DayOfWeek_Sin'] = np.sin(2 * np.pi * data_copy['DayOfWeek'] / 7)
    data_copy['DayOfWeek_Cos'] = np.cos(2 * np.pi * data_copy['DayOfWeek'] / 7)
    data_copy = data_copy.drop(columns=['Date'])
    return data_copy


def save_data(data, file_name):
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    data.to_csv(os.path.join('processed_data', file_name), index=False)


def plot_training_target_distribution(y_train):
    plt.hist(y_train, bins=30, alpha=0.5, color='g')
    plt.title('Distribution of Target Variable in Training Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def data_preparation(data, target_col_name, n_past=60, skip_feature_extraction_for=None):
    commodity_name = data['Commodity'].iloc[0]

    if skip_feature_extraction_for is not None and commodity_name in skip_feature_extraction_for:
        print(f"Skipping feature extraction for {commodity_name}")
    else:
        data = extract_date_features(data)

    data = data.drop(columns=['Commodity'])

    encoder = None
    if 'Unit' in data.columns:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        unit_encoded = encoder.fit_transform(data[['Unit']])
        unit_encoded_df = pd.DataFrame(unit_encoded, columns=[f"Unit_{cat}" for cat in encoder.categories_[0][1:]])
        data = pd.concat([data, unit_encoded_df], axis=1)
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

    if X.size == 0 or y.size == 0:
        print(f"No sequences generated for {commodity_name}. Skipping...")
        return np.array([]), np.array([]), np.array([]), np.array([]), None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler, encoder


# Main Script
data_path = "dataset.csv"
data = load_data(data_path)
commodities = data['Commodity'].unique()

for commodity in commodities:
    print(f"\nProcessing: {commodity}")
    commodity_data = data[data['Commodity'] == commodity]

    X_train, X_test, y_train, y_test, scaler, encoder = data_preparation(
        commodity_data,
        target_col_name='Average',
        n_past=60,
        skip_feature_extraction_for=None
    )

    if X_train.size == 0 or X_test.size == 0 or y_train.size == 0 or y_test.size == 0:
        print(f"Skipping further processing for {commodity} due to insufficient data.")
        continue

    plot_training_target_distribution(y_train)