import pandas as pd

# Load data
df = pd.read_csv('dataset.csv')

# Delete rows where 'Maize' appears anywhere
df = df[~df.isin(['Sweet Lime']).any(axis=1)]

# Save the modified data back to the file
df.to_csv('dataset.csv', index=False)