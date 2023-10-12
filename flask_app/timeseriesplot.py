import matplotlib.pyplot as plt
import pandas as pd

# Assume `data` DataFrame has columns: 'Date', 'Actual_Price' and 'Predicted_Price'
# Also assuming that Date is in 'yyyy-mm-dd' format

# Sample data
data = {
    'Date': pd.date_range(start="2013-01-01", periods=100, freq='M'),
    'Actual_Price': [x for x in range(100)],
    'Predicted_Price': [x*1.05 for x in range(100)]
}

df = pd.DataFrame(data)

# Creating a plot
plt.figure(figsize=(10, 6))

# Plotting actual prices
plt.plot(df['Date'], df['Actual_Price'], label='Actual Prices', marker='o')

# Plotting predicted prices
plt.plot(df['Date'], df['Predicted_Price'], label='Predicted Prices', linestyle='dashed', marker='x')

# Adding labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Commodity Price')
plt.title('Actual vs. Predicted Commodity Prices Over Time')
plt.legend()

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()
