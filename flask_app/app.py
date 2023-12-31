from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import logging
from tensorflow.keras.models import load_model
import pandas as pd


app = Flask(__name__)
df = pd.read_csv(r'C:\Users\suyog\PycharmProjects\pythonProject3\flask_app\dataset.csv')
logging.basicConfig(level=logging.INFO)

# Load model and scalers
model = load_model('saved_data/lstm_model.h5')
scaler_X = joblib.load('saved_data/scaler_X.pkl')
scaler_y = joblib.load('saved_data/scaler_y.pkl')
commodity_names = joblib.load('saved_data/commodity_names.pkl')
data = pd.read_csv(r'C:\Users\suyog\PycharmProjects\pythonProject3\flask_app\dataset.csv')

EXPECTED_FEATURE_COUNT = scaler_X.n_features_in_


@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict_page", methods=['GET', 'POST'])
def predict_page():
    return render_template("index.html", commodities=commodity_names)

@app.route('/api/commodities')
def get_commodities():
    df = pd.read_csv(r'C:\Users\suyog\PycharmProjects\pythonProject3\flask_app\dataset.csv')
    commodities = df['Commodity'].unique().tolist()
    return jsonify(commodities)

@app.route("/analyze")
def analyze():
    commodities = data['Commodity'].unique().tolist()
    return render_template('analyze.html', commodities=commodities)

@app.route("/commodity_data/<commodity_name>")
def get_commodity_data(commodity_name):
    specific_data = data[data['Commodity'] == commodity_name]
    return jsonify({
        "dates": specific_data['Date'].astype(str).tolist(),
        "min_prices": specific_data['Minimum'].tolist(),
        "max_prices": specific_data['Maximum'].tolist()
    })

@app.route("/search_commodity")
def search_commodity():
    term = request.args.get('term')
    search_results = [com for com in commodity_names if term.lower() in com.lower()]
    return jsonify(search_results)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Get the data from the form
        date = request.form['date']  # Format: 'YYYY-MM-DD'
        year, month, day = map(float, date.split('-'))
        commodity = request.form['commodity']

        # Debug: Check commodity count and model input shape
        logging.info(f"Total commodities: {len(commodity_names)}")
        logging.info(f"Model input shape: {model.input_shape[2]}")
        logging.info(f"One-hot vector size: {len(generate_one_hot_vector(commodity, commodity_names))}")

        # One-hot encode the commodity
        commodity_data = generate_one_hot_vector(commodity, commodity_names)

        # Debug: Check feature counts and commodity data
        logging.info(f"Commodity Input:  {commodity}")
        logging.info(f"One-hot Encoded:  {commodity_data}")
        logging.info(f"Expected Features by Scaler:  {EXPECTED_FEATURE_COUNT}")
        logging.info(f"Provided Feature Count:  {len(commodity_data) + 3}")

        # Quick fix: if features are less, pad them
        feature_count = len(commodity_data) + 3
        if feature_count < EXPECTED_FEATURE_COUNT:
            logging.warning("PADDING INPUT: Input features less than expected. Adding dummy features.")
            commodity_data.extend([0] * (EXPECTED_FEATURE_COUNT - feature_count))

        # Verify feature count
        assert len(
            commodity_data) + 3 == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(commodity_data) + 3}"

        # Prepare and scale the input data
        input_data = np.array([year, month, day] + commodity_data).reshape(1, -1)

        # Debug: Check shaped input data
        logging.info(f"Input Data:  {input_data}")

        input_scaled = scaler_X.transform(input_data)
        input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_original = scaler_y.inverse_transform(prediction).flatten()

        result = {
            'min_price': prediction_original[0],
            'max_price': prediction_original[1]
        }

        # Send additional data to the template
        return render_template("result.html", result=result, commodity_name=commodity, date=date)
    else:
        return "Error: Method not allowed"



def generate_one_hot_vector(input_category, all_possible_categories):
    return [1 if cat == input_category else 0 for cat in all_possible_categories]


if __name__ == "__main__":
    app.run(debug=True)

