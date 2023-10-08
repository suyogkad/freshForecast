import os

# Get the list of all files in the directory.
files = os.listdir(r"C:\Users\suyog\PycharmProjects\freshForecast\processed_data")

# Open the file `file.txt` for writing.
with open(r"C:\Users\suyog\PycharmProjects\freshForecast\file.txt", "w") as f:
    # Iterate through the list of files and write the file name to the file.
    for file in files:
        f.write(file + "\n")

# Close the file.
f.close()
