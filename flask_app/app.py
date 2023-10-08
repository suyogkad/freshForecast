from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Pre-load all models & scalers
models = {}
scalers = {}
# Use absolute paths for your models and scalers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the absolute path to your script
models_folder = os.path.join(BASE_DIR, 'models')
scalers_folder = os.path.join(BASE_DIR, 'scalers_encoders')

for model_file in os.listdir(models_folder):
    model_filepath = os.path.join(models_folder, model_file)
    if os.path.exists(model_filepath) and os.path.getsize(model_filepath) > 0:
        # File exists and is not empty
        print(f"Loading model: {model_filepath}")
        model_name = model_file.replace('_model.h5', '')
        models[model_name] = load_model(model_filepath)
        scalers[model_name] = joblib.load(os.path.join(scalers_folder, model_name + '_scaler.gz'))
    else:
        # File doesn't exist or is empty
        print(f"File: {model_filepath} doesn't exist or is empty")


# Load dataset
dataset = pd.read_csv('dataset.csv')


@app.route('/')
def home():
    # Send unique Commodity names to the template
    commodities = dataset['Commodity'].unique()
    return render_template('index.html', commodities=commodities)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        commodity = request.form['commodity']
        # ... other inputs

        # Identify model & scaler name
        model_name = commodity.lower().replace('(', '').replace(')', '').replace('/', '').replace(' ', '')

        # Load appropriate model and scaler
        model = models.get(model_name)
        scaler = scalers.get(model_name)

        if model and scaler:
            # Data preprocessing (scaling, reshaping, etc.)
            # Make sure the input data shape/formats match with the ones during training

            # Example: Dummy data for prediction
            # input_data = np.array([some_preprocessed_data])
            # input_data_reshaped = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

            # Make prediction
            prediction = model.predict(input_data_reshaped)

            # Inverse scaling (if applied during training)
            prediction_original_scale = scaler.inverse_transform(prediction)

            return render_template('prediction.html', prediction=prediction_original_scale)
        else:
            error_msg = f"No model or scaler found for: {commodity}"
            return render_template('index.html', error=error_msg, commodities=dataset['Commodity'].unique())

    return render_template('index.html', commodities=dataset['Commodity'].unique())


if __name__ == "__main__":
    app.run(debug=True)
