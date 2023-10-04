import pandas as pd

# load the data and ensure it's in the correct format
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    return data

if __name__ == "__main__":
    file_path = "dataset.csv"  # Replace with your actual file path
    data = load_data(file_path)
    print(data.head())
