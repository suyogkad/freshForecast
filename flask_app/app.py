import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import keras.models
import pickle
import gzip
from datetime import datetime

app = Flask(__name__)

# Load dataset to get the vegetable names
data = pd.read_csv('dataset.csv')
vege_names = sorted(data['Commodity'].unique())


# Sanitize name function
def sanitize_name(name):
    return name.replace(" ", "").replace("(", "").replace(")", "").lower()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        vege = request.form.get('vegetable')
        date = request.form.get('date')

        # Ensure the date selected is a future date
        selected_date = datetime.strptime(date, '%Y-%m-%d')
        if selected_date <= datetime.now():
            error = "Please select a future date."
            return render_template('index.html', vegetables=vege_names, error=error)

        return redirect(url_for('predict', vegetable=vege, date=date))

    return render_template('index.html', vegetables=vege_names, error=None)


@app.route('/predict/<vegetable>/<date>')
def predict(vegetable, date):
    sanitized_vege_name = sanitize_name(vegetable)

    model_name = sanitized_vege_name + "_model.h5"
    model = keras.models.load_model(os.path.join('models', model_name))

    # File path for scalers and encoders
    scaler_file_path = f'scalers_encoders/{sanitized_vege_name}_scaler.gz'
    encoder_file_path = f'scalers_encoders/{sanitized_vege_name}_encoder.gz'

    # Check if scaler and encoder files exist
    assert os.path.exists(scaler_file_path), f"Scaler file does not exist at path: {scaler_file_path}"
    assert os.path.exists(encoder_file_path), f"Encoder file does not exist at path: {encoder_file_path}"

    # Load scaler and encoder
    with gzip.open(scaler_file_path, 'rb') as f:
        scaler = pickle.load(f)

    with gzip.open(encoder_file_path, 'rb') as f:
        encoder = pickle.load(f)

    # Find the index (encoded value) of the provided date-string
    transformed_date = np.where(encoder == date)[0]

    # Ensure a match was found
    if transformed_date.size == 0:
        raise ValueError(f"No encoding found for date: {date}")

    # Convert to 2D array as model input
    transformed_date = transformed_date.reshape(1, -1)

    # Predict
    prediction = model.predict(scaler.transform(transformed_date))
    min_price, max_price = prediction[0]

    return render_template('results.html', vegetable=vegetable, date=date, min_price=min_price, max_price=max_price)


if __name__ == '__main__':
    app.run(debug=True)
