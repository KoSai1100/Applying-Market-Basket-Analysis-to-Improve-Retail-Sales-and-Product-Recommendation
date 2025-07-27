# CP2-SAM Web Application

This project is a web application that implements three different machine learning models: Random Forest, LSTM, and Apriori for market basket analysis. The application allows users to input data, run analyses, and view results through a user-friendly interface.

## Project Structure

```
cp2-sam-webapp
├── src
│   ├── app.py                # Main entry point of the web application
│   ├── models
│   │   ├── random_forest.py  # Implementation of the Random Forest model
│   │   ├── lstm.py           # Implementation of the LSTM model
│   │   └── apriori.py        # Implementation of the Apriori algorithm
│   ├── static
│   │   └── style.css         # CSS styles for the web application
│   ├── templates
│   │   ├── index.html        # Main HTML template for user input
│   │   ├── results.html      # Template for displaying results
│   │   └── comparison.html    # Template for model comparison
│   └── utils
│       └── data_processing.py # Utility functions for data processing
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd cp2-sam-webapp
   ```

2. **Install dependencies:**
   It is recommended to create a virtual environment before installing the dependencies.
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```
   python src/app.py
   ```

4. **Access the application:**
   Open your web browser and go to `http://127.0.0.1:5000`.

## Features

- **Random Forest Model:** Train and evaluate a Random Forest model for customer segmentation.
- **LSTM Model:** Forecast sales using an LSTM model based on historical data.
- **Apriori Algorithm:** Perform market basket analysis to find association rules between products.
- **Model Comparison:** View a comparison of the performance metrics of the different models.

## Usage Guidelines

- Navigate to the main page to input your data or select options for analysis.
- After submitting your input, results will be displayed on the results page.
- You can view a comparison of the models' performance metrics on the comparison page.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.