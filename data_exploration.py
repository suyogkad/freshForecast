import matplotlib.pyplot as plt
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    return data

def basic_statistics(data):
    print("\nBasic Statistics:\n", data.describe())
    print("\nMissing Values:\n", data.isnull().sum())

def visualize_trends(data, commodity):
    plt.figure(figsize=(10, 6))
    commodity_data = data[data['Commodity'] == commodity]
    plt.plot(commodity_data['Date'], commodity_data['Average'], label='Average Price')
    plt.title(f'Price Trend of {commodity}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "dataset.csv"  # our dataset path
    data = load_data(file_path)
    basic_statistics(data)

    commodity = "Tomato Big(Nepali)"  # using a commodity for exploration
    visualize_trends(data, commodity)
