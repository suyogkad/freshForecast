![freshForecast Logo](assets/template1.png)

Explore the financial horizons of Kathmandu's vital agricultural hub with freshForecast. Dive into a technological ensemble of rich historical data and acute machine learning precision, crafting not only a forecast but a tangible guide through the ebbs and flows of the Kalimati market's future prices.
## Description

freshForecast is an insightful deep learning project that casts light on the intricate price dynamics of the prominent Kalimati Fruits and Vegetables Market, which significantly satisfies 60-70% of Kathmandu Valley's demand for agricultural produce. This project harnesses the predictive power of Long Short-Term Memory (LSTM) networks— a specialized form of Recurrent Neural Networks (RNNs) — to analyze and forecast fruit and vegetable prices, utilizing a comprehensive dataset spanning from 2013 to 2021. The ambition here is not merely to visualize historical data but to adeptly predict future market prices, providing a vital tool for researchers, market analysts, and policymakers in crafting informed strategies and decisions in the agricultural market domain.

### Dataset

The dataset can be found here: [Kalimati Tarkari Dataset](https://opendatanepal.com/dataset/kalimati-tarkari-dataset)

## Getting Started

### Prerequisites

Ensure you have Python installed on your machine. If not, download and install it from [Python's official site](https://www.python.org/).

### Installation & Running

Follow these steps to get the project up and running on your local machine:

1. **Clone the Repository**

   ```sh
   git clone https://github.com/suyogkad/freshForecast.git
   cd freshForecast

2. **Install Requirements**

   Make sure to create a virtual environment before installing dependencies.
   
   ```sh
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  # Windows

   pip install -r requirements.txt
   
3. **Run the Flask app**
   
   ```sh
   python app.py
   ```
   
   Now, navigate to http://127.0.0.1:5000/ in your browser to access the application.

## Usage

Users can leverage freshForecast to meticulously analyze the fruits and vegetables prices from 2013 to 2021 in Kalimati, Kathmandu, and also to predict future prices. By selecting a specific fruit or vegetable, users can gain insights into its price predictions, aiding in more informed decision-making or deriving valuable insights for research and study purposes.

## License

This project is open source and available under the [MIT License](LICENSE) (Note: You might need to add a LICENSE file).
