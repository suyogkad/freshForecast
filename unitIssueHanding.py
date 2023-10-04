import pandas as pd
import numpy as np


def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    return data


def extract_date_features(data):
    data.loc[:, 'Year'] = data['Date'].dt.year
    data.loc[:, 'Month'] = data['Date'].dt.month
    data.loc[:, 'Day'] = data['Date'].dt.day
    data.loc[:, 'DayOfWeek'] = data['Date'].dt.dayofweek
    data.loc[:, 'Month_Sin'] = np.sin((data['Month'] - 1) * (2. * np.pi / 12))
    data.loc[:, 'Month_Cos'] = np.cos((data['Month'] - 1) * (2. * np.pi / 12))
    data.loc[:, 'Day_Sin'] = np.sin((data['Day'] - 1) * (2. * np.pi / 30))
    data.loc[:, 'Day_Cos'] = np.cos((data['Day'] - 1) * (2. * np.pi / 30))
    data.loc[:, 'DayOfWeek_Sin'] = np.sin(data['DayOfWeek'] * (2. * np.pi / 7))
    data.loc[:, 'DayOfWeek_Cos'] = np.cos(data['DayOfWeek'] * (2. * np.pi / 7))
    data = data.drop(columns=['Date'])
    return data


def generate_sequences(data, n_past=60):
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def data_preparation(data, target_col_name, n_past=60):
    data = extract_date_features(data)
    cols = [target_col_name] + [col for col in data if col != target_col_name]
    data = data[cols]

    non_numeric_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_cols:
        print(f"Warning: Non-numeric columns found: {non_numeric_cols}. These will be dropped.")
        data = data.drop(columns=non_numeric_cols)

    # Check data after dropping non-numeric columns
    print("\nData after dropping non-numeric columns:")
    print(data.head())
    print("Data shape: ", data.shape)

    # Check if any row is all NaN values after transformation
    print("\nRows with all NaN values:")
    print(data[data.isna().all(axis=1)])

    data = data.dropna()  # Drop NaN rows if any

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Check scaled data
    print("\nScaled data:")
    print(data_scaled[:5])
    print("Scaled data shape: ", data_scaled.shape)

    X, y = generate_sequences(data_scaled, n_past)

    # Check sequences
    print("\nGenerated sequences (X and y):")
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    return X, y


if __name__ == "__main__":
    file_path = "dataset.csv"
    data = load_data(file_path)

    maize_data = data[data['Commodity'] == 'Maize'].copy()
    print("Maize data:")
    print(maize_data.head())
    print("Maize data shape: ", maize_data.shape)

    X, y = data_preparation(maize_data, target_col_name='Average', n_past=60)
