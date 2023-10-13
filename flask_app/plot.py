import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
import joblib

# Load dataset
data = pd.read_csv('dataset.csv')

# Define the columnst to scale for X and Y data
x_columns_to_scale = ['Minimum', 'Maximum']
y_columns_to_scale = ['Minimum', 'Maximum']

# Select columns that exist in dataset for scaling
x_columns_to_scale = [col for col in x_columns_to_scale if col in data.columns]
y_columns_to_scale = [col for col in y_columns_to_scale if col in data.columns]

# Create custom scalers for X and Y data
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Fit the scalers only on the selected columns in dataset
scaler_x.fit(data[x_columns_to_scale])
scaler_y.fit(data[y_columns_to_scale])

# Apply the scalers to data for X and Y data
data_scaled_x = scaler_x.transform(data[x_columns_to_scale]) if x_columns_to_scale else data
data_scaled_y = scaler_y.transform(data[y_columns_to_scale]) if y_columns_to_scale else data

# Create a DataFrame for the scaled data
data_scaled = pd.DataFrame({**{f'Scaled_{col}': data_scaled_x[:, i] for i, col in enumerate(x_columns_to_scale)},
                            **{f'Scaled_{col}': data_scaled_y[:, i] for i, col in enumerate(y_columns_to_scale)}})

# Create a pairplot
sns.pairplot(data_scaled)

# Display the plot
plt.show()
